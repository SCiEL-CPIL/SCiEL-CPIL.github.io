    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    for u in range(n * m):
        j = u % m
        U[u][j] = 1
        
    for v in range(m * p):
        j = v // p
        V[v][j] = 1
