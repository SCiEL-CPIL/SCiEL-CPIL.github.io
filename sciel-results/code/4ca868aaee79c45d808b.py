def construct():
    n, m, p = DIMENSIONS
    rank = n
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        # U[i*m + j][i] = 1 for all j
        for j in range(m):
            U[i * m + j][i] = 1
        # V[j*p + k][i] = 1 for all j, k
        for j in range(m):
            for k in range(p):
                V[j * p + k][i] = 1
        # W[k*n + i][i] = 1 for all k
        for k in range(p):
            W[k * n + i][i] = 1
            
    return {"U": U, "V": V, "W": W}
