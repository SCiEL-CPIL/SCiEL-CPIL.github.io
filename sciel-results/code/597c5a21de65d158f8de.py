def construct():
    n, m, p = 2, 3, 3
    rank = m  # 3
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][j] = 1
    
    for j in range(m):
        for k in range(p):
            V[j * p + k][j] = 1
    
    for k in range(p):
        for i in range(n):
            for c in range(rank):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
