def construct():
    n, m, p = 2, 3, 4
    rank = 12
    
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    for k in range(p):
        for d in range(m):  # d = 0, 1, 2
            c = k * m + d
            # W[k*2+i, c] = 1 for i=0,1
            for i in range(n):
                W[k*n+i][c] = 1.0
            # V[j*4+k, c] = 1 if j==d else 0
            for j in range(m):
                if j == d:
                    V[j*p+k][c] = 1.0
            # U[i*3+j, c] = 1 if j==d else 0
            for i in range(n):
                for j in range(m):
                    if j == d:
                        U[i*m+j][c] = 1.0
    
    return {"U": U, "V": V, "W": W}
