def _build_4x1x6():
    n, m, p = 4, 1, 6
    rank = 1
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for i in range(n*m):
        U[i][0] = 1
    for i in range(m*p):
        V[i][0] = 1
    for i in range(p*n):
        W[i][0] = 1
        
    return U, V, W
