def construct():
    n, m, p = 3, 3, 8
    rank = 36
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # Block 1: k=0,1,2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    # Block 2: k=3,4,5
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i*m+j][col] = 1
                V[j*p+(k+3)][col] = 1
                W[(k+3)*n+i][col] = 1
                col += 1
    # Block 3: k=6,7
    for i in range(n):
        for j in range(m):
            for k in range(2):
                U[i*m+j][col] = 1
                V[j*p+(k+6)][col] = 1
                W[(k+6)*n+i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
