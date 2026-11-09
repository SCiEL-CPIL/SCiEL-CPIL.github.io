def construct():
    n, m, p = DIMENSIONS
    rank = m  # 3
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill U and V
    for i in range(n):
        for j in range(m):
            U[i * m + j][j] = 1
            # V depends on j and k
            for k in range(p):
                V[j * p + k][j] = 1
                
    # Fill W
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
