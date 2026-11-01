def construct():
    n, m, p = 2, 2, 15
    # 7 blocks of size 2, 1 block of size 1
    num_blocks_2 = 7
    num_blocks_1 = 1
    rank = num_blocks_2 * 7 + num_blocks_1 * 4
    rank = 53
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53
    
    col = 0
    # 7 Strassen blocks
    for b in range(7):
        k0 = 2 * b
        # Strassen terms
        # Term 1: M1 = (A00+A11)(B00+B11)
        U[0][col] = 1; U[3][col] = 1
        V[k0][col] = 1; V[15+k0+1][col] = 1
        W[2*k0][col] = 1; W[2*k0+3][col] = 1
        col += 1
        
        # Term 2: M2 = (A10+A11)B00
        U[2][col] = 1; U[3][col] = 1
        V[k0][col] = 1
        W[2*k0][col] = 1; W[2*k0+2][col] = 1
        col += 1
        
        # Term 3: M3 = A00(B01-B11)
        U[0][col] = 1
        V[k0+1][col] = 1; V[15+k0+1][col] = -1
        W[2*k0+1][col] = 1; W[2*k0+3][col] = 1
        col += 1
        
        # Term 4: M4 = A11(B10-B00)
        U[3][col] = 1
        V[15+k0][col] = 1; V[k0][col] = -1
        W[2*k0+2][col] = 1; W[2*k0+3][col] = 1
        col += 1
        
        # Term 5: M5 = (A00+A01)B11
        U[0][col] = 1; U[1][col] = 1
        V[15+k0+1][col] = 1
        W[2*k0+1][col] = 1; W[2*k0+3][col] = -1
        col += 1
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        U[2][col] = 1; U[0][col] = -1
        V[k0][col] = 1; V[k0+1][col] = 1
        W[2*k0+2][col] = 1
        col += 1
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        U[1][col] = 1; U[3][col] = -1
        V[15+k0][col] = 1; V[15+k0+1][col] = 1
        W[2*k0+3][col] = 1
        col += 1
    
    # 1 schoolbook block for column 14
    k0 = 14
    for j in range(2):
        for i in range(2):
            U[i*2+j][col] = 1
            V[j*15+k0][col] = 1
            W[k0*2+i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
