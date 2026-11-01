def construct():
    n, m, p = 2, 3, 8
    rank = m  # = 3
    U = [[0] * rank for _ in range(n * m)]  # 6 x 3
    V = [[0] * rank for _ in range(m * p)]  # 24 x 3
    W = [[0] * rank for _ in range(p * n)]  # 16 x 3
    
    for j in range(m):
        for i in range(n):
            U[i * m + j][j] = 1
        for k in range(p):
            V[j * p + k][j] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i][j] = 1
    
    return {"U": U, "V": V, "W": W}
