def construct():
    n, m, p = DIMENSIONS
    U_4x4, V_4x4, W_4x4 = get_strassen_4x4_factors()
    
    # Indices for non-zero elements in the padded 4x4 matrices
    # A' (2x3 in top-left of 4x4): indices 0,1,2,4,5,6
    idx_A = [0, 1, 2, 4, 5, 6]
    # B' (3x3 in top-left of 4x4): indices 0,1,2,4,5,6,8,9,10
    idx_B = [0, 1, 2, 4, 5, 6, 8, 9, 10]
    # C' (2x3 in top-left of 4x4): indices 0,1,2,4,5,6
    idx_C = [0, 1, 2, 4, 5, 6]
    
    U_2x3x3 = U_4x4[idx_A, :]
    V_2x3x3 = V_4x4[idx_B, :]
    W_2x3x3 = W_4x4[idx_C, :]
    
    total_rank = 42
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    for block in range(3):
        offset = block * 14
        U[:, offset:offset+14] = U_2x3x3
        
        # V mapping
        for i in range(3):
            for j in range(3):
                row_full = i * 9 + (3 * block + j)
                row_block = i * 3 + j
                V[row_full, offset:offset+14] = V_2x3x3[row_block, :]
        
        # W mapping
        for i in range(2):
            for j in range(3):
                row_full = (3 * block + j) * 2 + i
                row_block = i * 3 + j
                W[row_full, offset:offset+14] = W_2x3x3[row_block, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
