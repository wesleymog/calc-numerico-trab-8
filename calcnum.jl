using LinearAlgebra

export minquad, euler, vandermonde, bissecao, newton, ponto_fixo

function minquad(x, y, F)
    """
    Minimos Quadrados
    - Recebe os pontos (x, y) e um vetor com os termos do polinomio/funcao F
    - Retorna os coeficientes da funcao que melhor aproxima os pontos (x,y) no formato F
    """
    # criar matriz zerada com o n° de linhas = n° de pontos (x,y) e
    # numero de colunas = numero de termos da funcao F
    V = zeros(length(x), length(F))
    
    # preenche a matriz de vandermonde
    for linha = 1:length(x)
        for coluna = 1:length(F)
            V[linha, coluna] = F[coluna](x[linha])
        end
    end
    
    # VᵀVc = Vᵀy
    Vᵀ = transpose(V)
    return Vᵀ * V \ Vᵀ * y
end


# Método de Euler (PVI)
# recebe x0 (inicial), xf(final), y0(inicial) e a derivada f.
# h é o intervalo entre cada x novo
function euler(x0, xf, y0, f, h = 0.1)
    N = (xf - x0) / h  # numero de passos
    y = y0
    x = x0
    for i = 1:N
        y = y + h * f(x, y)
        x = x + h
    end
    return x, y
end


# Recebe os pontos (x, y)
# Retorna os coeficientes da interpolacao polinomial dos pontos (x, y)
function vandermonde(x, y)
    # criar matriz zerada com as dimensoes certas
    V = zeros(length(x), length(y))
    
    # preencher matriz de vandermonde
    for linha = 1:length(x)
        for coluna = 1:length(x)
            V[linha, coluna] = x[linha]^(coluna - 1)
        end
    end
    
    if isa(y, Matrix)
        y = transpose(y)
    end
    
    return V\y
end


# Método da Bisseção
"""
Tenta achar um zero para a funcao dentro de um intervalo dado.
Entrada: funcao f, extremo a, extremo b, 
            (tolerância absoluta atol), (tolerância relativa rtol),
            (tempo maximo max_time), (quantidade maxima de iteracoes max_iter)
Retorna: x, fx, exitflag
"""

function bissecao(f, a, b;
                  atol = 1e-8, rtol = 1e-8,
                  max_time = 10.0, max_iter = 1000,
                  )
    fa = f(a)
    fb = f(b)
    
    # calcula as tolerâncias
    # tolerância do valor da funcao
    ϵ = atol + rtol * max(abs(fa), abs(fb))
    #println("Bissecao - ϵ = $(ϵ)")
    # tolerancia do tamanho do intervalo
    ϵba = atol + rtol * abs(b - a)
    #println("Bissecao - ϵba = $(ϵ)")
    
    # verifica condiçoes iniciais
    if abs(fa) ≤ ϵ
        return a, fa, :sucesso
    elseif abs(fb) ≤ ϵ
        return b, fb, :sucesso
    elseif fa * fb ≥ 0
        return a, fa, :falha_sinais_opostos
    end
    
    # calcula o ponto médio
    x = (a + b) / 2
    fx = f(x)
    
    # define variaveis de tempo e qtd de iteraçoes
    iter = 0
    t0 = time()
    Δt = time() - t0
    
    # condicao para o algoritmo ter resolvido o problema
    resolvido = (abs(fx) ≤ ϵ || abs(b - a) ≤ ϵba)
    
    # condicao para o algoritmo pedir arrego
    cansado = (iter ≥ max_iter || Δt ≥ max_time)
    
    while !(resolvido || cansado)
        # Verifica qual seçao vamos escolher:
        # se a e x tem sinais ≠, pega a 1ª seçao
        if fa * fx < 0
            b = x
            fb = fx
        # senao, pega a 2ª seçao
        else
            a = x
            fa = fx
        end
        
        # calcula o ponto médio
        x = (a + b) / 2
        fx = f(x)
        
        # calcula condicoes para o while
        iter = iter + 1
        Δt = time() - t0
        resolvido = (abs(fx) ≤ ϵ || abs(b - a) ≤ ϵba)
        cansado = (iter ≥ max_iter || Δt ≥ max_time)
    end
    
    println("Iterações = $(iter)")
    println("Tempo = $(Δt)")
    println("Tamanho do intervalo = $(abs(b - a))")
    
    exitflag = :desconhecido
    if resolvido
        exitflag = :sucesso
    elseif cansado
        if iter ≥ max_iter
            exitflag = :max_iter
        else
            exitflag = :max_time
        end
    end
    
    println("Valor de x = $x")
    println("Valor de f(x) = $fx")
    
    return x, fx, exitflag
end


# Método de Newton
"""
Tenta achar um zero para a funcao, a partir da funcao, da derivada da funcao e de um valor inicial para x.
Existem chutes bons e ruins para x e nao é possivel saber de antemao se o método vai convergir ou nao.
Entrada: funcao f, derivada da funcao fd, valor inicial x, 
            (tolerância absoluta atol), (tolerância relativa rtol),
            (tempo maximo max_time), (quantidade maxima de iteracoes max_iter)
Retorna: x, fx, exitflag
"""

function newton(f, fd, x;
                  atol = 1e-8, rtol = 1e-8,
                  max_time = 10.0, max_iter = 1000,
                  )
    
    fx = f(x)
    
    # tolerância do valor da funcao
    ϵ = atol + rtol * abs(fx)
    #println("Newton - ϵ = $(ϵ)")
    
    # define variaveis de tempo e qtd de iteraçoes
    iter = 0
    t0 = time()
    Δt = time() - t0
    
    # caso o algoritmo nao termine de nenhuma maneira esperada
    exitflag = :desconhecido
    
    # condicao para o algoritmo ter resolvido o problema
    resolvido = (abs(fx) ≤ ϵ)
    
    # condicao para o algoritmo pedir arrego
    cansado = (iter ≥ max_iter || Δt ≥ max_time)
    
    while !(resolvido || cansado)
        # se a derivada for muito proxima de zero, o método vai divergir
        fdx = fd(x)
        if abs(fdx) ≤ ϵ
            exitflag = :derivada_nula
            break
        end
        
        # calcula os novos valores de x e f(x)
        x = x - fx / fdx 
        fx = f(x)
        
        # calcula condicoes para o while
        iter = iter + 1
        Δt = time() - t0
        resolvido = (abs(fx) ≤ ϵ)
        cansado = (iter ≥ max_iter || Δt ≥ max_time)
    end
    
    println("Iterações = $(iter)")
    println("Tempo = $(Δt)")
    
    if resolvido
        exitflag = :sucesso
    elseif cansado
        if iter ≥ max_iter
            exitflag = :max_iter
        else
            exitflag = :max_time
        end
    end
    
    #println("Valor de x = $x")
    #println("Valor de f(x) = $fx")
    
    return x, fx, exitflag
end


# Método de Ponto Fixo
"""
Tenta achar o ponto fixo de uma iteracao.
Entrada: funcao g(x), valor inicial x, 
            (tolerância absoluta atol), (tolerância relativa rtol),
            (tempo maximo max_time), (quantidade maxima de iteracoes max_iter)
Retorna: x, g(x), exitflag
"""

function ponto_fixo(g, x;
                  atol = 1e-8, rtol = 1e-8,
                  max_time = 10.0, max_iter = 1000,
                  )
    
    gx = g(x)
    
    # tolerância do valor da funcao
    ϵ = atol + rtol * abs(gx)
    println("ϵ = $(ϵ)")
    
    # define variaveis de tempo e qtd de iteraçoes
    iter = 0
    t0 = time()
    Δt = time() - t0
    
    # caso o algoritmo nao termine de nenhuma maneira esperada
    exitflag = :desconhecido
    
    # condicao para o algoritmo ter resolvido o problema
    resolvido = (abs(gx - x) ≤ ϵ)
    
    # condicao para o algoritmo pedir arrego
    cansado = (iter ≥ max_iter || Δt ≥ max_time)
    
    while !(resolvido || cansado)
        # calcula os novos valores de x e g(x)
        x = g(x) 
        gx = g(x)
        
        # calcula condicoes para o while
        iter = iter + 1
        Δt = time() - t0
        resolvido = (abs(gx - x) ≤ ϵ)
        cansado = (iter ≥ max_iter || Δt ≥ max_time)
    end
    
    println("Iterações = $(iter)")
    println("Tempo = $(Δt)")
    
    if resolvido
        exitflag = :sucesso
    elseif cansado
        if iter ≥ max_iter
            exitflag = :max_iter
        else
            exitflag = :max_time
        end
    end
    
    return x, gx, exitflag
end


""" Função auxiliar 
(Baseada na funcao compartilhada pelo Daniel Hashimoto no Classroom)
transforma uma lista de coeficientes em uma função

Ex de entrada:
ax^2 + bx + c => [ c, b, a ]
"""
function apply(coefs, x)
    grau = length(coefs)-1
    len = grau + 1
    accx = 1
    acc = 0
    for i = 1:len
        acc += coefs[i] * accx
        accx *= x
    end
    return acc
end

# simples (1 intervalo so)
function pontomedio(f, a, b)
    return (b - a) * f((a + b) / 2)
end

# simples (1 intervalo so)
function trapezio(f, a, b)
    return (b - a) * (f(a) + f(b)) / 2
end

# simples (1 intervalo so)
function simpson(f, a, b)
    return (b - a) * (f(a) + 4f((a + b) / 2) + f(b)) / 6
end    

function trapezio_composto(f, a, b, n)
    """
    Recebe uma função f, 2 números reais a e b e 
    1 número positivo n.
    
    Retorna um valor aproximado para a integral de f no intervalo de a até b, 
    usando n trapézios no método do trapézio composto.
    """
    
    # definir tamanho do intervalo
    h = (b - a) / n
    
    # soma dos valores de f(x) que se repetem na soma dos trapézios
    soma = 0.0
    for i = 1:n-1
        x = a + i * h
        soma += f(x)
    end
    
    soma = f(a) + 2*soma + f(b)
    
    return h * soma / 2
end

# versao alternativa usando a funcao do trapezio simples
function trapezio_composto2(f, a, b, n)
    h = (b - a) / n
    
    soma = 0.0
    for i = 1:n
        soma += trapezio(f, a, a + h)
        a += h
    end
    
    return soma
end

# Método do trapézio para integral dupla, simples (1 intervalo so)
function trapezoidal3D(f, a, b, c, d)
    return (d - c) * (b - a) * (f(d,b) + f(d,a) + f(c,b) + f(c,a)) / 4
end

function trapezoidal3D_composto(f, a, b, c, d, N)
    """
    Recebe uma funcao f(x,y), os reais a, b, c, d e o inteiro N.
    Vamos aproximar a integral dupla de f(x)dxdy.
    O intervalo para y é de a até b.
    O intervalo para x é de c até d.
    N é o numero de solidos trapezoidais que vamos somar para aproximar o valor da integral.
    """
    
    # tamanho dos intervalos para x
    hx = (d - c) / N
    # tamanho dos intervalos para y
    hy = (b - a) / N

    soma = 0.0
    for i = 1:N
        ci = c
        for j = 1:N
            soma += trapezoidal3D(f, a, a + hy, ci, ci + hx)
            ci += hx
        end
        a += hy
    end
    
    return soma
end