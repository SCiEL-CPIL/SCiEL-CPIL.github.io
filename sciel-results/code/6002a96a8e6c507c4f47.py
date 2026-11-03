def construct():
    n, m, p = 2, 3, 6
    rank = 12
    U = [[0]*rank for _ in range(n*m)]  # 6 x 12
    V = [[0]*rank for _ in range(m*p)]  # 18 x 12
    W = [[0]*rank for _ in range(p*n)]  # 12 x 12
    
    for k in range(p):
        for i in range(n):
            t = k*n + i  # column index
            W[k*n+i][t] = 1
            for j in range(m):
                U[i*m+j][t] = 1
                V[j*p+k][t] = 1
    
    return {"U": U, "V": V, "W": W}
