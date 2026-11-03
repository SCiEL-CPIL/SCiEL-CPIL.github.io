def construct():
    n, m, p = 2, 5, 6
    rank = m * p  # 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            U[j][r] = 1
            U[m + j][r] = 1
            V[j * p + k][r] = 1
            W[k * n][r] = 1
            W[k * n + 1][r] = 1
    
    return {"U": U, "V": V, "W": W}
