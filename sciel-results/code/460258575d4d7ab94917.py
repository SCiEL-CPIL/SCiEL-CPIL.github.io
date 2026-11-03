def construct():
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part
    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1
        # Define the 7 terms
        # s=0
        col = 7 * g + 0
        U[0][col] = 1; U[4][col] = 1
        V[k0][col] = 1; V[8+k1][col] = 1
        W[2*k0][col] = 1; W[2*k1+1][col] = 1
        
        # s=1
        col = 7 * g + 1
        U[3][col] = 1; U[4][col] = 1
        V[k0][col] = 1
        W[2*k0+1][col] = 1; W[2*k1+1][col] = -1
        
        # s=2
        col = 7 * g + 2
        U[0][col] = 1
        V[8+k0][col] = 1
        W[2*k1][col] = 1; W[2*k1+1][col] = 1
        
        # s=3
        col = 7 * g + 3
        U[4][col] = 1
        V[8+k1][col] = 1
        W[2*k0][col] = 1; W[2*k0+1][col] = 1
        
        # s=4
        col = 7 * g + 4
        U[0][col] = 1; U[1][col] = 1
        V[8+k1][col] = 1
        W[2*k0][col] = -1; W[2*k1][col] = 1
        
        # s=5
        col = 7 * g + 5
        U[3][col] = 1; U[0][col] = -1
        V[k0][col] = 1; V[k1][col] = 1
        W[2*k1+1][col] = 1
        
        # s=6
        col = 7 * g + 6
        U[1][col] = 1; U[4][col] = -1
        V[k1][col] = 1; V[8+k1][col] = 1
        W[2*k0][col] = 1

    # Direct part
    base_col = 28
    for i in range(2):
        for k in range(8):
            col = base_col + i * 8 + k
            U[i * 3 + 2][col] = 1
            V[16 + k][col] = 1
            W[k * 2 + i][col] = 1

    return {"U": U, "V": V, "W": W}
