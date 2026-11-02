def construct():
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[:,0] (2x1) * B[0,:] (1x8) -> 16 terms
    for i in range(2):
        for k in range(8):
            U[i * 3][col] = 1
            V[k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    
    # Part 2: A[:,1:3] (2x2) * B[1:3,:] (2x8) -> 28 terms (4 Strassen blocks)
    for block in range(4):
        b0 = 8 + block * 2
        b1 = 8 + block * 2 + 1
        b2 = 16 + block * 2
        b3 = 16 + block * 2 + 1
        c00 = block * 4
        c01 = block * 4 + 1
        c10 = block * 4 + 2
        c11 = block * 4 + 3
        
        U[2][col] = 1; U[5][col] = 1
        V[b0][col] = 1; V[b3][col] = 1
        W[c00][col] = 1; W[c11][col] = 1
        col += 1
        
        U[4][col] = 1; U[5][col] = 1
        V[b0][col] = 1
        W[c10][col] = 1; W[c11][col] = -1
        col += 1
        
        U[2][col] = 1
        V[b1][col] = 1; V[b3][col] = -1
        W[c01][col] = 1; W[c11][col] = 1
        col += 1
        
        U[5][col] = 1
        V[b2][col] = 1; V[b0][col] = -1
        W[c00][col] = 1; W[c10][col] = 1
        col += 1
        
        U[2][col] = 1; U[3][col] = 1
        V[b3][col] = 1
        W[c00][col] = -1; W[c01][col] = 1
        col += 1
        
        U[4][col] = 1; U[2][col] = -1
        V[b0][col] = 1; V[b1][col] = 1
        W[c11][col] = 1
        col += 1
        
        U[3][col] = 1; U[5][col] = -1
        V[b2][col] = 1; V[b3][col] = 1
        W[c00][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
