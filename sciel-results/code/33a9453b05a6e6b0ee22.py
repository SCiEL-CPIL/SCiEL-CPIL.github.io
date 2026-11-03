def construct():
    n, m, p = 3, 5, 6
    rank = m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        # U: index i*m + j, column j is 1 for all i
        for i in range(n):
            U[i * m + j][j] = 1
        # V: index j*p + k, column j is 1 for all k
        for k in range(p):
            V[j * p + k][j] = 1
        # W: index k*n + i, column j is 1 for all k, i
        for k in range(p):
            for i in range(n):
                W[k * n + i][j] = 1
                
    return {"U": U, "V": V, "W": W}
