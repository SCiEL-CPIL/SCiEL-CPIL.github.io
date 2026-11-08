def construct():
    n, m, p = 2, 4, 6
    rank = m * p  # 24
    U = [[0] * rank for _ in range(n * m)]  # 8 x 24
    V = [[0] * rank for _ in range(m * p)]  # 24 x 24
    W = [[0] * rank for _ in range(p * n)]  # 12 x 24
    
    col = 0
    for j in range(m):
        for k in range(p):
            # U: rows i*m+j for i=0,1
            U[0 * m + j][col] = 1
            U[1 * m + j][col] = 1
            # V: row j*p+k
            V[j * p + k][col] = 1
            # W: rows k*n+i for i=0,1
            W[k * n + 0][col] = 1
            W[k * n + 1][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
