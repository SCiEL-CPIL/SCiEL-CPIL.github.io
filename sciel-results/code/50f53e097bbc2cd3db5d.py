import numpy as np

def construct():
    n, m, p = 4, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 16, 20, 20
    
    # We will construct a rank 65 decomposition.
    # 49 terms for the 4x4x4 part (Strassen)
    # 16 terms for the 4x4x1 part (Trivial)
    
    rank = 65
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Helper to add a rank-1 term
    def add_term(u_vec, v_vec, w_vec, col_idx):
        U[:, col_idx] = u_vec
        V[:, col_idx] = v_vec
        W[:, col_idx] = w_vec

    # --- Part 1: 4x4x4 Strassen Decomposition (Rank 49) ---
    
    # Strassen 2x2 decomposition vectors (local indices 0..3 for A, B, C)
    # A: 00, 01, 10, 11 -> indices 0,1,2,3
    # B: 00, 01, 10, 11 -> indices 0,1,2,3
    # C: 00, 01, 10, 11 -> indices 0,1,2,3
    
    strassen_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    strassen_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1],# M3
        [0, 1, -1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    strassen_w = [
        [1, 0, 0, 1], # M1 -> C00+, C11+
        [0, 0, 1, -1],# M2 -> C10+, C11-
        [0, 1, 0, 0], # M3 -> C01+
        [1, 0, 1, 0], # M4 -> C00+, C10+
        [-1, 1, 0, 0],# M5 -> C00-, C01+
        [0, 0, 0, 1], # M6 -> C11+
        [-1, 0, 0, -1]# M7 -> C00-, C11-
    ]
    
    # Define the 7 block products S1..S7
    # Each S is defined by (P_blocks, Q_blocks, C_contrib)
    # P_blocks: dict {block_idx: coeff}
    # Q_blocks: dict {block_idx: coeff}
    # C_contrib: dict {c_block_idx: coeff}
    
    # Block indices:
    # A: 0:00, 1:01, 2:10, 3:11
    # B: 0:00, 1:01, 2:10, 3:11
    # C: 0:00, 1:01, 2:10, 3:11
    
    S_defs = [
        # S1 = (A00 + A11) (B00 + B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # S2 = (A10 + A11) B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # S3 = A00 (B10 - B11)
        ({0: 1}, {2: 1, 3: -1}, {1: 1}),
        # S4 = A11 (B01 - B00)
        ({3: 1}, {1: 1, 0: -1}, {0: 1, 2: 1}),
        # S5 = (A00 + A01) B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # S6 = (A10 - A00) (B00 + B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # S7 = (A01 - A11) (B10 + B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: -1, 3: -1})
    ]
    
    col_idx = 0
    for S_idx, (p_blocks, q_blocks, c_contrib) in enumerate(S_defs):
        # For each of the 7 Strassen scalar steps
        for s_idx in range(7):
            u_loc = strassen_u[s_idx]
            v_loc = strassen_v[s_idx]
            w_loc = strassen_w[s_idx]
            
            u_glob = np.zeros(nm)
            v_glob = np.zeros(mp)
            w_glob = np.zeros(pn)
            
            # Expand u_loc to u_glob
            # u_loc[0] corresponds to P00, u_loc[1] to P01, etc.
            # P00 is sum of A blocks in p_blocks with their coeffs
            # The local index 0 (P00) maps to A block 0 (00), 1 (01), 2 (10), 3 (11) in the 2x2 block structure?
            # No, P is a 2x2 matrix of blocks? No, P is a 2x2 matrix of scalars (linear forms).
            # The Strassen step uses P as a 2x2 matrix.
            # P[0,0] is a linear form. P[0,1] is a linear form.
            # u_loc[0] multiplies P[0,0].
            
            # We need to map the linear form P[r,c] to global A indices.
            # P[r,c] = sum_{b_idx, coeff in p_blocks} coeff * A_block[b_idx][r,c]
            
            # A_block[b_idx] has indices:
            # b=0 (00): A[0,0], A[0,1], A[1,0], A[1,1] -> global 0,1,4,5?
            # Wait, A is 4x4.
            # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
            # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
            # A[2,0]=8, A[2,1]=9, A[2,2]=10, A[2,3]=11
            # A[3,0]=12, A[3,1]=13, A[3,2]=14, A[3,3]=15
            
            # Block 0 (A00): rows 0-1, cols 0-1. Indices: 0,1,4,5.
            # Block 1 (A01): rows 0-1, cols 2-3. Indices: 2,3,6,7.
            # Block 2 (A10): rows 2-3, cols 0-1. Indices: 8,9,12,13.
            # Block 3 (A11): rows 2-3, cols 2-3. Indices: 10,11,14,15.
            
            a_block_indices = {
                0: [0, 1, 4, 5],
                1: [2, 3, 6, 7],
                2: [8, 9, 12, 13],
                3: [10, 11, 14, 15]
            }
            
            # Similarly for B
            # B is 4x4.
            # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3
            # B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7
            # B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11
            # B[3,0]=12, B[3,1]=13, B[3,2]=14, B[3,3]=15
            
            b_block_indices = {
                0: [0, 1, 4, 5],
                1: [2, 3, 6, 7],
                2: [8, 9, 12, 13],
                3: [10, 11, 14, 15]
            }
            
            # For each local entry (r,c) in the 2x2 P matrix
            # u_loc[r*2+c] is the coefficient.
            # P[r,c] = sum_{b, coeff} coeff * A_block[b][r,c]
            # A_block[b][r,c] index in block: r*2+c (if block is 2x2)
            # Global index: a_block_indices[b][r*2+c]
            
            for r in range(2):
                for c in range(2):
                    coeff_u = u_loc[r*2+c]
                    if coeff_u != 0:
                        for b_idx, p_coeff in p_blocks.items():
                            val = coeff_u * p_coeff
                            g_idx = a_block_indices[b_idx][r*2+c]
                            u_glob[g_idx] += val
            
            # Similarly for V
            for r in range(2):
                for c in range(2):
                    coeff_v = v_loc[r*2+c]
                    if coeff_v != 0:
                        for b_idx, q_coeff in q_blocks.items():
                            val = coeff_v * q_coeff
                            g_idx = b_block_indices[b_idx][r*2+c]
                            v_glob[g_idx] += val
            
            # Similarly for W
            # w_loc[r*2+c] is coefficient for C_block[r,c] in the local 2x2 result.
            # This local result S is added to global C blocks defined in c_contrib.
            # c_contrib: {c_block_idx: coeff}
            # For a local C entry (r,c), it corresponds to C_block[rb, cb] in global?
            # No, S is a 2x2 matrix.
            # S[r,c] is added to global C blocks.
            # Which global C block does S[r,c] go to?
            # S is a block in the 2x2 block matrix C.
            # S is at position (S_row, S_col) in the block matrix?
            # No, S1..S7 are just intermediate 2x2 matrices.
            # They are summed to form the 4 2x2 blocks of C.
            # c_contrib tells us which C blocks receive S and with what weight.
            # But S is a 2x2 matrix.
            # So S[r,c] is added to C_block[rb, cb][r,c] for all (rb,cb) in c_contrib?
            # No, c_contrib maps C_block_index to weight.
            # If c_contrib = {0: 1, 3: 1}, it means S is added to C00 and C11.
            # So S[r,c] adds to C00[r,c] and C11[r,c].
            
            # C_block indices:
            # 0: C00 (rows 0-1, cols 0-1). Global indices: 0,1,4,5.
            # 1: C01 (rows 0-1, cols 2-3). Global indices: 2,3,6,7.
            # 2: C10 (rows 2-3, cols 0-1). Global indices: 8,9,12,13.
            # 3: C11 (rows 2-3, cols 2-3). Global indices: 10,11,14,15.
            
            c_block_indices = {
                0: [0, 1, 4, 5],
                1: [2, 3, 6, 7],
                2: [8, 9, 12, 13],
                3: [10, 11, 14, 15]
            }
            
            for r in range(2):
                for c in range(2):
                    coeff_w = w_loc[r*2+c]
                    if coeff_w != 0:
                        for c_idx, c_coeff in c_contrib.items():
                            val = coeff_w * c_coeff
                            g_idx = c_block_indices[c_idx][r*2+c]
                            w_glob[g_idx] += val
            
            add_term(u_glob, v_glob, w_glob, col_idx)
            col_idx += 1
            
    # --- Part 2: 4x4x1 Trivial Decomposition (Rank 16) ---
    # For k=4 (5th column of B)
    # A is 4x4. B_col is 4x1. C_col is 4x1.
    # Terms: A[i,j] * B[j,4] -> C[4,i]
    # a = i*4 + j
    # b = j*4 + 4
    # c = 4*4 + i = 16 + i
    
    for i in range(4):
        for j in range(4):
            u_glob = np.zeros(nm)
            v_glob = np.zeros(mp)
            w_glob = np.zeros(pn)
            
            a_idx = i * 4 + j
            b_idx = j * 4 + 4
            c_idx = 16 + i
            
            u_glob[a_idx] = 1
            v_glob[b_idx] = 1
            w_glob[c_idx] = 1
            
            add_term(u_glob, v_glob, w_glob, col_idx)
            col_idx += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
