def construct():
    n, m, p = 2, 4, 8
    rank = 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        # U[i*m + j, r] = 1 if j == r
        for i in range(n):
            U[i * m + r][r] = 1
        # V[j*p + k, r] = 1 if j == r
        for k in range(p):
            V[r * p + k][r] = 1
        # W[k*n + i, r] = 1 for all k, i
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
