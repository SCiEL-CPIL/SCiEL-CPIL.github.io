def construct():
    n, m, p = 2, 3, 9
    rank = m  # 3
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for j in range(m):
        # U: index i*m + j is 1 for all i
        for i in range(n):
            U[i * m + j][j] = 1.0
            
        # V: index j*p + k is 1 for all k
        for k in range(p):
            V[j * p + k][j] = 1.0
            
        # W: index k*n + i is 1 for all k, i
        for k in range(p):
            for i in range(n):
                W[k * n + i][j] = 1.0

    return {"U": U, "V": V, "W": W}
