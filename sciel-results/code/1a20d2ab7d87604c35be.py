def construct():
    n, m, p = 2, 3, 10
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                U[idx_u][r] = 1
                V[idx_v][r] = 1
                W[idx_w][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}
