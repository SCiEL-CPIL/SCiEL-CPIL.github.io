def construct():
    n, m, p = 2, 3, 10
    rank = m * p  # 30
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # V is identity
    for r in range(rank):
        V[r][r] = 1
    
    # Set U and W
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = j * p + k
                U[i*m + j][col] = 1
                W[k*n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
