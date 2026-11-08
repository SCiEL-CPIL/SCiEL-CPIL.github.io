def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = m * p  # 16
    U = [[0]*rank for _ in range(n*m)]  # 8×16
    V = [[0]*rank for _ in range(m*p)]  # 16×16
    W = [[0]*rank for _ in range(p*n)]  # 8×16
    
    r = 0
    for j in range(m):
        for k in range(p):
            # U: for i in range(n), U[i*m+j, r] = 1
            for i in range(n):
                U[i*m+j][r] = 1
            # V: V[j*p+k, r] = 1
            V[j*p+k][r] = 1
            # W: for i in range(n), W[k*n+i, r] = 1
            for i in range(n):
                W[k*n+i][r] = 1
            r += 1
    
    return {"U": U, "V": V, "W": W}
