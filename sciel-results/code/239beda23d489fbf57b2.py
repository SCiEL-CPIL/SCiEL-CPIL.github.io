"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    
    # 2x2 Strassen decomposition
    # U2, V2, W2 are 4x7
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, -1, -1]
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
    
    # Verify 2x2 Strassen
    # C = A B
    # A, B, C are 2x2
    # U2 @ diag(A) ... no, U2 maps A to linear combos.
    # Let's just trust the standard Strassen matrices.
    
    # 4x4 Strassen
    # 7 blocks M1..M7
    # Each Mk is 2x2
    # Total rank 49
    
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Define block indices for A, B, C
    # A is 4x4, blocks A00, A01, A10, A11
    # A00: (0,0), (0,1), (1,0), (1,1)
    # A01: (0,2), (0,3), (1,2), (1,3)
    # A10: (2,0), (2,1), (3,0), (3,1)
    # A11: (2,2), (2,3), (3,2), (3,3)
    
    # Helper to get indices for a block
    def block_indices(block_row, block_col):
        # block_row, block_col in {0, 1}
        # Returns list of 4 indices for the 2x2 block
        # Row offset = block_row * 2
        # Col offset = block_col * 2
        r_off = block_row * 2
        c_off = block_col * 2
        return [r_off*4 + c_off, r_off*4 + c_off + 1, (r_off+1)*4 + c_off, (r_off+1)*4 + c_off + 1]
    
    # Mk definitions
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Mk, we use 2x2 Strassen
    # Mk = Xk Yk
    # Xk is linear combo of A blocks
    # Yk is linear combo of B blocks
    
    # We will iterate over the 7 Mk's
    for k in range(7):
        col_start = k * 7
        
        # Determine Xk and Yk coefficients
        # Xk is 2x2, Yk is 2x2
        # Xk = sum_{br, bc} coeff_A[br, bc] A_{br, bc}
        # Yk = sum_{br, bc} coeff_B[br, bc] B_{br, bc}
        
        coeff_A = np.zeros((2, 2, 2, 2)) # [br, bc, r, c] for 2x2 block
        coeff_B = np.zeros((2, 2, 2, 2))
        
        if k == 0: # M1
            # X = A00 + A11
            coeff_A[0, 0] = np.eye(2)
            coeff_A[1, 1] = np.eye(2)
            # Y = B00 + B11
            coeff_B[0, 0] = np.eye(2)
            coeff_B[1, 1] = np.eye(2)
        elif k == 1: # M2
            # X = A10 + A11
            coeff_A[1, 0] = np.eye(2)
            coeff_A[1, 1] = np.eye(2)
            # Y = B00
            coeff_B[0, 0] = np.eye(2)
        elif k == 2: # M3
            # X = A00
            coeff_A[0, 0] = np.eye(2)
            # Y = B01 - B11
            coeff_B[0, 1] = np.eye(2)
            coeff_B[1, 1] = -np.eye(2)
        elif k == 3: # M4
            # X = A11
            coeff_A[1, 1] = np.eye(2)
            # Y = B10 - B00
            coeff_B[1, 0] = np.eye(2)
            coeff_B[0, 0] = -np.eye(2)
        elif k == 4: # M5
            # X = A00 + A01
            coeff_A[0, 0] = np.eye(2)
            coeff_A[0, 1] = np.eye(2)
            # Y = B11
            coeff_B[1, 1] = np.eye(2)
        elif k == 5: # M6
            # X = A10 - A00
            coeff_A[1, 0] = np.eye(2)
            coeff_A[0, 0] = -np.eye(2)
            # Y = B00 + B01
            coeff_B[0, 0] = np.eye(2)
            coeff_B[0, 1] = np.eye(2)
        elif k == 6: # M7
            # X = A01 - A11
            coeff_A[0, 1] = np.eye(2)
            coeff_A[1, 1] = -np.eye(2)
            # Y = B10 + B11
            coeff_B[1, 0] = np.eye(2)
            coeff_B[1, 1] = np.eye(2)
            
        # Now construct U, V, W for this Mk
        # For each of the 7 columns in 2x2 Strassen
        for c2 in range(7):
            col = col_start + c2
            
            # U part
            # U2[c2] gives coefficients for X00, X01, X10, X11
            # X00 = sum coeff_A[0,0] A00
            # So U[:, col] += U2[0, c2] * (coeff_A[0,0] flattened to A indices)
            for br in range(2):
                for bc in range(2):
                    # coeff_A[br, bc] is 2x2
                    # U2 row 0 corresponds to X00 (br=0, bc=0)
                    # U2 row 1 corresponds to X01 (br=0, bc=1)
                    # U2 row 2 corresponds to X10 (br=1, bc=0)
                    # U2 row 3 corresponds to X11 (br=1, bc=1)
                    u2_val = U2[br*2 + bc, c2]
                    if u2_val != 0:
                        indices = block_indices(br, bc)
                        for idx, val in zip(indices, coeff_A[br, bc].flatten()):
                            U[idx, col] += u2_val * val
            
            # V part
            for br in range(2):
                for bc in range(2):
                    v2_val = V2[br*2 + bc, c2]
                    if v2_val != 0:
                        indices = block_indices(br, bc)
                        for idx, val in zip(indices, coeff_B[br, bc].flatten()):
                            V[idx, col] += v2_val * val
            
            # W part
            # Mk contributes to C blocks with certain coefficients
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # W2 rows correspond to Mk's 00, 01, 10, 11 entries
            # These map to C blocks
            # Mk is a 2x2 matrix.
            # Mk[0,0] contributes to C00[0,0], etc.
            
            # Coefficients of Mk in C00, C01, C10, C11
            coeff_C = np.zeros((2, 2)) # [block_row, block_col]
            
            if k == 0: # M1
                coeff_C[0, 0] = 1
                coeff_C[1, 1] = 1
            elif k == 1: # M2
                coeff_C[1, 0] = 1
                coeff_C[1, 1] = -1
            elif k == 2: # M3
                coeff_C[0, 1] = 1
                coeff_C[1, 1] = 1
            elif k == 3: # M4
                coeff_C[0, 0] = 1
                coeff_C[1, 0] = 1
            elif k == 4: # M5
                coeff_C[0, 0] = -1
                coeff_C[0, 1] = 1
            elif k == 5: # M6
                coeff_C[1, 1] = 1
            elif k == 6: # M7
                coeff_C[0, 0] = 1
            
            for br in range(2):
                for bc in range(2):
                    c_coeff = coeff_C[br, bc]
                    if c_coeff != 0:
                        # W2 row 0 -> Mk[0,0] -> C[br, bc][0,0]
                        # W2 row 1 -> Mk[0,1] -> C[br, bc][0,1]
                        # W2 row 2 -> Mk[1,0] -> C[br, bc][1,0]
                        # W2 row 3 -> Mk[1,1] -> C[br, bc][1,1]
                        for r2 in range(2):
                            for c2_ in range(2):
                                w2_val = W2[r2*2 + c2_, c2]
                                if w2_val != 0:
                                    indices = block_indices(br, bc)
                                    # indices are [0, 1, 2, 3] for 00, 01, 10, 11
                                    # r2*2 + c2_ corresponds to index in 2x2
                                    # 00->0, 01->1, 10->2, 11->3
                                    idx = indices[r2*2 + c2_]
                                    W[idx, col] += c_coeff * w2_val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
