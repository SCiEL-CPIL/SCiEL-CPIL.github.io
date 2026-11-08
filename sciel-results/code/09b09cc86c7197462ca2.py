import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition matrices
    # U2, V2 are 4x7, W2 is 4x7
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Top-level Strassen coefficients for 4x4 (2x2 blocks)
    # alpha[i, j, k] is the coefficient of P_k in C_block_ij
    # i, j in {0, 1} for blocks
    # k in 0..6 for P_1..P_7
    alpha = np.zeros((2, 2, 7))
    # C_11 = P_1 + P_4 - P_5 + P_7
    alpha[0, 0, 0] = 1
    alpha[0, 0, 3] = 1
    alpha[0, 0, 4] = -1
    alpha[0, 0, 6] = 1
    # C_12 = P_3 + P_5
    alpha[0, 1, 2] = 1
    alpha[0, 1, 4] = 1
    # C_21 = P_2 + P_4
    alpha[1, 0, 1] = 1
    alpha[1, 0, 3] = 1
    # C_22 = P_1 - P_2 + P_3 + P_6
    alpha[1, 1, 0] = 1
    alpha[1, 1, 1] = -1
    alpha[1, 1, 2] = 1
    alpha[1, 1, 5] = 1
    
    # Define block extraction matrices L_A and L_B
    # A is 4x4, flattened to 16.
    # Blocks A_00 (0,0), A_01 (0,1), A_10 (1,0), A_11 (1,1)
    # Indices:
    # A_00: 0,1,4,5
    # A_01: 2,3,6,7
    # A_10: 8,9,12,13
    # A_11: 10,11,14,15
    
    # L_A[k] is 4x16, extracts the 2x2 matrix X_k (flattened)
    # X_1 = A_00 + A_11
    # X_2 = A_10 + A_11
    # X_3 = A_00
    # X_4 = A_11
    # X_5 = A_00 + A_01
    # X_6 = A_10 - A_00
    # X_7 = A_01 - A_11
    
    def get_block_indices(bi, bj):
        # bi, bj in 0..1
        # Returns list of 4 indices in the 16-vector
        # Row range: bi*2 to bi*2+1
        # Col range: bj*2 to bj*2+1
        idxs = []
        for r in range(2):
            for c in range(2):
                global_r = bi * 2 + r
                global_c = bj * 2 + c
                idxs.append(global_r * 4 + global_c)
        return idxs

    block_idxs = {
        (0,0): get_block_indices(0,0),
        (0,1): get_block_indices(0,1),
        (1,0): get_block_indices(1,0),
        (1,1): get_block_indices(1,1)
    }
    
    # Construct L_A (7 matrices of 4x16)
    L_A = []
    for k in range(7):
        L = np.zeros((4, 16))
        # Map X_k blocks to L
        # X_k is 2x2. We need to fill L such that L @ a = flattened X_k
        # The order of entries in flattened X_k is 00, 01, 10, 11
        # L[row, col] = 1 if col is in block_idxs[bi,bj] and row corresponds to that position
        
        # Helper to add block
        def add_block(L, bi, bj, sign):
            idxs = block_idxs[(bi,bj)]
            for r in range(2):
                for c in range(2):
                    local_idx = r * 2 + c
                    global_idx = idxs[local_idx]
                    L[local_idx, global_idx] += sign
                    
        if k == 0: # X_1 = A_00 + A_11
            add_block(L, 0, 0, 1)
            add_block(L, 1, 1, 1)
        elif k == 1: # X_2 = A_10 + A_11
            add_block(L, 1, 0, 1)
            add_block(L, 1, 1, 1)
        elif k == 2: # X_3 = A_00
            add_block(L, 0, 0, 1)
        elif k == 3: # X_4 = A_11
            add_block(L, 1, 1, 1)
        elif k == 4: # X_5 = A_00 + A_01
            add_block(L, 0, 0, 1)
            add_block(L, 0, 1, 1)
        elif k == 5: # X_6 = A_10 - A_00
            add_block(L, 1, 0, 1)
            add_block(L, 0, 0, -1)
        elif k == 6: # X_7 = A_01 - A_11
            add_block(L, 0, 1, 1)
            add_block(L, 1, 1, -1)
            
        L_A.append(L)
        
    # Construct L_B (7 matrices of 4x16)
    # Y_1 = B_00 + B_11
    # Y_2 = B_00
    # Y_3 = B_01 - B_11
    # Y_4 = B_10 - B_01
    # Y_5 = B_11
    # Y_6 = B_00 + B_01
    # Y_7 = B_10 + B_11
    
    L_B = []
    for k in range(7):
        L = np.zeros((4, 16))
        def add_block(L, bi, bj, sign):
            idxs = block_idxs[(bi,bj)]
            for r in range(2):
                for c in range(2):
                    local_idx = r * 2 + c
                    global_idx = idxs[local_idx]
                    L[local_idx, global_idx] += sign
                    
        if k == 0: # Y_1
            add_block(L, 0, 0, 1)
            add_block(L, 1, 1, 1)
        elif k == 1: # Y_2
            add_block(L, 0, 0, 1)
        elif k == 2: # Y_3
            add_block(L, 0, 1, 1)
            add_block(L, 1, 1, -1)
        elif k == 3: # Y_4
            add_block(L, 1, 0, 1)
            add_block(L, 0, 1, -1)
        elif k == 4: # Y_5
            add_block(L, 1, 1, 1)
        elif k == 5: # Y_6
            add_block(L, 0, 0, 1)
            add_block(L, 0, 1, 1)
        elif k == 6: # Y_7
            add_block(L, 1, 0, 1)
            add_block(L, 1, 1, 1)
            
        L_B.append(L)

    # Construct U, V, W
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    for k in range(7):
        for j in range(7):
            col = k * 7 + j
            # U col
            U[:, col] = L_A[k].T @ U2[:, j]
            # V col
            V[:, col] = L_B[k].T @ V2[:, j]
            
            # W col
            # W[global_idx, col] = sum over blocks (i,j) and local rows r?
            # No, global_idx corresponds to a specific block (i,j) and local pos r.
            # We can just iterate over all 16 global indices.
            for g in range(16):
                # Determine block (i,j) and local pos r for g
                # g = row * 4 + col
                row = g // 4
                col_g = g % 4
                bi = row // 2
                bj = col_g // 2
                r = row % 2
                c = col_g % 2
                local_idx = r * 2 + c
                
                coeff = alpha[bi, bj, k] * W2[local_idx, j]
                W[g, col] = coeff
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
