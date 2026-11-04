import numpy as np

def _strassen_2x2_decomposition():
    """
    Returns U, V, W for 2x2 * 2x2 -> 2x2 Strassen decomposition (rank 7).
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a11+a22)*(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 3] = 1
    
    # M2 = (a21+a22)*b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 3] = -1
    
    # M3 = a11*(b12-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 1] = 1; W[3, 3] = 1
    
    # M4 = a22*(b21-b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 0] = 1; W[2, 1] = 1
    
    # M5 = (a11+a12)*b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 0] = -1; W[0, 1] = 1
    
    # M6 = (a21-a11)*(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 3] = 1
    
    # M7 = (a12-a22)*(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 0] = 1
    
    return U, V, W

def construct():
    n, m, p = 4, 4, 7
    
    # Rank calculation:
    # 4x4 * 4x4 part: Strassen rank 49
    # 4x4 * 4x3 part: Schoolbook rank 48
    # Total rank = 97
    
    rank = 97
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # --- Part 1: 4x4 * 4x4 (Strassen) ---
    U_b, V_b, W_b = _strassen_2x2_decomposition()
    
    # Define block indices for A, B, C (4x4 matrices)
    # Block (r, c) of 2x2 blocks corresponds to rows 2r, 2r+1 and cols 2c, 2c+1
    # Indices in flat 4x4 matrix (row-major):
    # Block 0 (top-left): 0, 1, 4, 5
    # Block 1 (top-right): 2, 3, 6, 7
    # Block 2 (bot-left): 8, 9, 12, 13
    # Block 3 (bot-right): 10, 11, 14, 15
    
    def get_block_indices(r, c):
        # r, c in {0, 1}
        start_row = 2 * r
        start_col = 2 * c
        return [
            start_row * 4 + start_col,
            start_row * 4 + start_col + 1,
            (start_row + 1) * 4 + start_col,
            (start_row + 1) * 4 + start_col + 1
        ]
    
    # Coefficient matrices for A' and B' for each of the 7 outer terms
    # CA[k] is 4x16, CB[k] is 4x16
    CA = []
    CB = []
    
    for k in range(7):
        ca = np.zeros((4, 16))
        cb = np.zeros((4, 16))
        
        if k == 0: # M1
            # A' = A11 + A22
            idx_11 = get_block_indices(0, 0)
            idx_22 = get_block_indices(1, 1)
            for i in range(4):
                ca[i, idx_11[i]] = 1
                ca[i, idx_22[i]] = 1
            # B' = B11 + B22
            cb = ca.copy()
        elif k == 1: # M2
            # A' = A21 + A22
            idx_21 = get_block_indices(1, 0)
            idx_22 = get_block_indices(1, 1)
            for i in range(4):
                ca[i, idx_21[i]] = 1
                ca[i, idx_22[i]] = 1
            # B' = B11
            idx_11 = get_block_indices(0, 0)
            for i in range(4):
                cb[i, idx_11[i]] = 1
        elif k == 2: # M3
            # A' = A11
            idx_11 = get_block_indices(0, 0)
            for i in range(4):
                ca[i, idx_11[i]] = 1
            # B' = B12 - B22
            idx_12 = get_block_indices(0, 1)
            idx_22 = get_block_indices(1, 1)
            for i in range(4):
                cb[i, idx_12[i]] = 1
                cb[i, idx_22[i]] = -1
        elif k == 3: # M4
            # A' = A22
            idx_22 = get_block_indices(1, 1)
            for i in range(4):
                ca[i, idx_22[i]] = 1
            # B' = B21 - B11
            idx_21 = get_block_indices(1, 0)
            idx_11 = get_block_indices(0, 0)
            for i in range(4):
                cb[i, idx_21[i]] = 1
                cb[i, idx_11[i]] = -1
        elif k == 4: # M5
            # A' = A11 + A12
            idx_11 = get_block_indices(0, 0)
            idx_12 = get_block_indices(0, 1)
            for i in range(4):
                ca[i, idx_11[i]] = 1
                ca[i, idx_12[i]] = 1
            # B' = B22
            idx_22 = get_block_indices(1, 1)
            for i in range(4):
                cb[i, idx_22[i]] = 1
        elif k == 5: # M6
            # A' = A21 - A11
            idx_21 = get_block_indices(1, 0)
            idx_11 = get_block_indices(0, 0)
            for i in range(4):
                ca[i, idx_21[i]] = 1
                ca[i, idx_11[i]] = -1
            # B' = B11 + B12
            idx_11 = get_block_indices(0, 0)
            idx_12 = get_block_indices(0, 1)
            for i in range(4):
                cb[i, idx_11[i]] = 1
                cb[i, idx_12[i]] = 1
        elif k == 6: # M7
            # A' = A12 - A22
            idx_12 = get_block_indices(0, 1)
            idx_22 = get_block_indices(1, 1)
            for i in range(4):
                ca[i, idx_12[i]] = 1
                ca[i, idx_22[i]] = -1
            # B' = B21 + B22
            idx_21 = get_block_indices(1, 0)
            idx_22 = get_block_indices(1, 1)
            for i in range(4):
                cb[i, idx_21[i]] = 1
                cb[i, idx_22[i]] = 1
        
        CA.append(ca)
        CB.append(cb)
        
    # Coefficient matrix for C mapping
    # CW[k] is 4x16, mapping M_k entries to C entries
    CW = []
    for k in range(7):
        cw = np.zeros((4, 16))
        if k == 0: # M1 -> C11 (+1), C22 (+1)
            idx_11 = get_block_indices(0, 0)
            idx_22 = get_block_indices(1, 1)
            for i in range(4):
                cw[i, idx_11[i]] = 1
                cw[i, idx_22[i]] = 1
        elif k == 1: # M2 -> C21 (+1), C22 (-1)
            idx_21 = get_block_indices(1, 0)
            idx_22 = get_block_indices(1, 1)
            for i in range(4):
                cw[i, idx_21[i]] = 1
                cw[i, idx_22[i]] = -1
        elif k == 2: # M3 -> C12 (+1), C22 (+1)
            idx_12 = get_block_indices(0, 1)
            idx_22 = get_block_indices(1, 1)
            for i in range(4):
                cw[i, idx_12[i]] = 1
                cw[i, idx_22[i]] = 1
        elif k == 3: # M4 -> C11 (+1), C21 (+1)
            idx_11 = get_block_indices(0, 0)
            idx_21 = get_block_indices(1, 0)
            for i in range(4):
                cw[i, idx_11[i]] = 1
                cw[i, idx_21[i]] = 1
        elif k == 4: # M5 -> C11 (-1), C12 (+1)
            idx_11 = get_block_indices(0, 0)
            idx_12 = get_block_indices(0, 1)
            for i in range(4):
                cw[i, idx_11[i]] = -1
                cw[i, idx_12[i]] = 1
        elif k == 5: # M6 -> C22 (+1)
            idx_22 = get_block_indices(1, 1)
            for i in range(4):
                cw[i, idx_22[i]] = 1
        elif k == 6: # M7 -> C11 (+1)
            idx_11 = get_block_indices(0, 0)
            for i in range(4):
                cw[i, idx_11[i]] = 1
        CW.append(cw)
        
    # Fill U, V, W for Part 1
    offset = 0
    for k in range(7):
        for l in range(7):
            col = offset + l
            U[:, col] = CA[k].T @ U_b[:, l]
            V[:, col] = CB[k].T @ V_b[:, l]
            W[:, col] = CW[k].T @ W_b[:, l]
    offset += 49
    
    # --- Part 2: 4x4 * 4x3 (Schoolbook) ---
    # A indices: 0..15
    # B indices for cols 4,5,6:
    # j=0, k=4,5,6 -> 4, 5, 6
    # j=1, k=4,5,6 -> 11, 12, 13
    # j=2, k=4,5,6 -> 18, 19, 20
    # j=3, k=4,5,6 -> 25, 26, 27
    # C indices for cols 4,5,6:
    # k=4, i=0..3 -> 16, 17, 18, 19
    # k=5, i=0..3 -> 20, 21, 22, 23
    # k=6, i=0..3 -> 24, 25, 26, 27
    
    col = offset
    for i in range(4):
        for j in range(4):
            for k in range(3):
                b_idx = j * 7 + (4 + k)
                c_idx = (4 + k) * 4 + i
                U[i * 4 + j, col] = 1
                V[b_idx, col] = 1
                W[c_idx, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
