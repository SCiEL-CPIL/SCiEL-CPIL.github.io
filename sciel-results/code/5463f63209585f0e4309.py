def construct():
    n, m, p = 2, 4, 6
    rank = 42
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_strassen_block(A_idxs, B_idxs, C_idxs, start_col):
        # A_idxs: [a11, a12, a21, a22] indices in U
        # B_idxs: [b11, b12, b21, b22] indices in V
        # C_idxs: [c11, c12, c21, c22] indices in W
        
        col = start_col
        # M1 = (a11+a22)(b11+b22)
        U[A_idxs[0]][col] += 1
        U[A_idxs[2]][col] += 1
        V[B_idxs[0]][col] += 1
        V[B_idxs[2]][col] += 1
        W[C_idxs[0]][col] += 1
        W[C_idxs[3]][col] += 1
        
        col += 1
        # M2 = (a21+a22)b11
        U[A_idxs[2]][col] += 1
        U[A_idxs[3]][col] += 1
        V[B_idxs[0]][col] += 1
        W[C_idxs[2]][col] += 1
        W[C_idxs[3]][col] += 1
        
        col += 1
        # M3 = a11(b12-b22)
        U[A_idxs[0]][col] += 1
        V[B_idxs[1]][col] += 1
        V[B_idxs[3]][col] += -1
        W[C_idxs[1]][col] += 1
        W[C_idxs[3]][col] += 1
        
        col += 1
        # M4 = a22(b21-b11)
        U[A_idxs[3]][col] += 1
        V[B_idxs[2]][col] += 1
        V[B_idxs[0]][col] += -1
        W[C_idxs[0]][col] += 1
        W[C_idxs[2]][col] += 1
        
        col += 1
        # M5 = (a11+a12)b22
        U[A_idxs[0]][col] += 1
        U[A_idxs[1]][col] += 1
        V[B_idxs[3]][col] += 1
        W[C_idxs[0]][col] += -1
        W[C_idxs[1]][col] += 1
        
        col += 1
        # M6 = (a21-a11)(b11+b12)
        U[A_idxs[2]][col] += 1
        U[A_idxs[0]][col] += -1
        V[B_idxs[0]][col] += 1
        V[B_idxs[1]][col] += 1
        W[C_idxs[3]][col] += 1
        
        col += 1
        # M7 = (a12-a22)(b21+b22)
        U[A_idxs[1]][col] += 1
        U[A_idxs[3]][col] += -1
        V[B_idxs[2]][col] += 1
        V[B_idxs[3]][col] += 1
        W[C_idxs[0]][col] += 1
        W[C_idxs[1]][col] += 1
        
        return col

    # A_L indices: [0, 1, 4, 5]
    # A_R indices: [2, 3, 6, 7]
    
    # B_T blocks:
    # B_T0: [0, 1, 6, 7]
    # B_T1: [2, 3, 8, 9]
    # B_T2: [4, 5, 10, 11]
    
    # B_B blocks:
    # B_B0: [12, 13, 18, 19]
    # B_B1: [14, 15, 20, 21]
    # B_B2: [16, 17, 22, 23]
    
    # C blocks:
    # C0: [0, 1, 2, 3]
    # C1: [4, 5, 6, 7]
    # C2: [8, 9, 10, 11]
    
    A_L = [0, 1, 4, 5]
    A_R = [2, 3, 6, 7]
    
    B_T0 = [0, 1, 6, 7]
    B_T1 = [2, 3, 8, 9]
    B_T2 = [4, 5, 10, 11]
    
    B_B0 = [12, 13, 18, 19]
    B_B1 = [14, 15, 20, 21]
    B_B2 = [16, 17, 22, 23]
    
    C0 = [0, 1, 2, 3]
    C1 = [4, 5, 6, 7]
    C2 = [8, 9, 10, 11]
    
    col = 0
    col = add_strassen_block(A_L, B_T0, C0, col)
    col = add_strassen_block(A_L, B_T1, C1, col)
    col = add_strassen_block(A_L, B_T2, C2, col)
    
    col = add_strassen_block(A_R, B_B0, C0, col)
    col = add_strassen_block(A_R, B_B1, C1, col)
    col = add_strassen_block(A_R, B_B2, C2, col)
    
    return {"U": U, "V": V, "W": W}
