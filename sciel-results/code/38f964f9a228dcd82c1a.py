def construct():
    n, m, p = 2, 4, 5
    rank = m  # = 4
    
    U = [[0]*rank for _ in range(n*m)]  # (8, 4)
    V = [[0]*rank for _ in range(m*p)]  # (20, 4)
    W = [[0]*rank for _ in range(p*n)]  # (10, 4)
    
    for c in range(rank):
        for i in range(n):
            U[i*m + c][c] = 1
        for k in range(p):
            V[c*p + k][c] = 1
        for i in range(n):
            for k in range(p):
                W[k*n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
