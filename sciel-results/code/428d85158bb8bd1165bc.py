import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 77 decomposition
    rank = 77
    
    # Initialize U, V, W as lists of lists of zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    # u_vec: length 16 (4x4)
    # v_vec: length 24 (4x6)
    # w_vec: length 24 (6x4)
    col = 0
    
    # --- Part 1: 4x4 by 4x4 (First 4 columns of B) ---
    # B indices for first 4 columns: j in 0..3, k in 0..3
    # C indices for first 4 columns: k in 0..3, i in 0..3
    
    # Strassen 2x2 by 2x2 base tensors
    # u, v, w are length 4 vectors corresponding to 2x2 matrices
    base_u = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0],# P6
        [0, 1, 0, -1] # P7
    ]
    base_v = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [-1, 0, 1, 0],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    base_w = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 0], # P2
        [0, 1, 0, 0], # P3
        [-1, 0, 1, 0],# P4
        [0, 1, 0, 1], # P5
        [1, 0, 1, 0], # P6
        [-1, 0, 0, -1]# P7
    ]
    
    # Strassen coefficients for 2x2 blocks of 4x4
    # M_k = S_k * T_k
    # S_k is linear combo of A blocks A1..A4
    # T_k is linear combo of B blocks B1..B4
    # C blocks are linear combos of M_k
    
    # A blocks indices in 4x4 (row-major):
    # A1: 0,1,4,5
    # A2: 2,3,6,7
    # A3: 8,9,12,13
    # A4: 10,11,14,15
    
    # B blocks indices in 4x4 part of 4x6 (row-major, k<4):
    # B1: 0,1,6,7
    # B2: 2,3,8,9
    # B3: 12,13,18,19
    # B4: 14,15,20,21
    
    # C blocks indices in 4x4 part of 6x4 (row-major, k<4):
    # C1: 0,1,4,5
    # C2: 2,3,6,7
    # C3: 8,9,12,13
    # C4: 10,11,14,15
    
    # Define block index mappings
    def get_block_indices(base_offset, block_id):
        # block_id 0..3
        # Returns list of 4 indices
        # 2x2 block in 4x4 matrix
        # r in 0,1, c in 0,1
        # global row = base_r + r, global col = base_c + c
        # index = row * width + col
        pass

    # We will iterate through the 7 Strassen steps for the 4x4 by 4x4 part
    # For each step k (0..6), we have a linear form in A and a linear form in B
    # We expand each linear form into the full 16-dim vector for U and 24-dim for V
    # Then we combine with the 7 base tensors for the 2x2 multiplication
    
    # Strassen S and T definitions
    # S1 = A1 + A4
    # S2 = A3 + A4
    # S3 = A1 + A2
    # S4 = A4 - A1
    # S5 = A1 + A3
    # S6 = A4 - A2
    # S7 = A2 - A4
    
    # T1 = B1 + B4
    # T2 = B1
    # T3 = B2 - B4
    # T4 = B3 - B1
    # T5 = B4
    # T6 = B1 + B2
    # T7 = B3 + B4
    
    # C1 = M1 - M4 + M6 - M7
    # C2 = M3 + M5
    # C3 = M2 + M4
    # C4 = M1 - M3 + M5 - M6
    
    # We need to construct U, V, W for each of the 49 terms.
    # Term (k, l) where k is Strassen step (0..6), l is base tensor (0..6)
    
    # Helper to add a vector to U, V, W
    def add_term(u_16, v_24, w_24):
        nonlocal col
        for i in range(16):
            if u_16[i] != 0:
                U[i][col] = u_16[i]
        for i in range(24):
            if v_24[i] != 0:
                V[i][col] = v_24[i]
        for i in range(24):
            if w_24[i] != 0:
                W[i][col] = w_24[i]
        col += 1
        
    # Precompute block indices for A, B, C
    # A is 4x4
    A_blocks = {
        1: [0, 1, 4, 5],
        2: [2, 3, 6, 7],
        3: [8, 9, 12, 13],
        4: [10, 11, 14, 15]
    }
    # B is 4x6, we only use first 4 cols for this part
    # B1: rows 0-1, cols 0-1 -> idx 0,1,6,7
    # B2: rows 0-1, cols 2-3 -> idx 2,3,8,9
    # B3: rows 2-3, cols 0-1 -> idx 12,13,18,19
    # B4: rows 2-3, cols 2-3 -> idx 14,15,20,21
    B_blocks = {
        1: [0, 1, 6, 7],
        2: [2, 3, 8, 9],
        3: [12, 13, 18, 19],
        4: [14, 15, 20, 21]
    }
    # C is 6x4, we only use first 4 rows for this part
    # C1: rows 0-1, cols 0-1 -> idx 0,1,4,5
    # C2: rows 0-1, cols 2-3 -> idx 2,3,6,7
    # C3: rows 2-3, cols 0-1 -> idx 8,9,12,13
    # C4: rows 2-3, cols 2-3 -> idx 10,11,14,15
    C_blocks = {
        1: [0, 1, 4, 5],
        2: [2, 3, 6, 7],
        3: [8, 9, 12, 13],
        4: [10, 11, 14, 15]
    }
    
    # Strassen S coefficients: dict block_id -> coeff
    S_coeffs = [
        {1: 1, 4: 1}, # S1
        {3: 1, 4: 1}, # S2
        {1: 1, 2: 1}, # S3
        {4: 1, 1: -1},# S4
        {1: 1, 3: 1}, # S5
        {4: 1, 2: -1},# S6
        {2: 1, 4: -1} # S7
    ]
    
    # Strassen T coefficients
    T_coeffs = [
        {1: 1, 4: 1}, # T1
        {1: 1},       # T2
        {2: 1, 4: -1},# T3
        {3: 1, 1: -1},# T4
        {4: 1},       # T5
        {1: 1, 2: 1}, # T6
        {3: 1, 4: 1}  # T7
    ]
    
    # C combination coefficients: C_block -> {M_k: coeff}
    C_coeffs = {
        1: {1: 1, 4: -1, 6: 1, 7: -1},
        2: {3: 1, 5: 1},
        3: {2: 1, 4: 1},
        4: {1: 1, 3: -1, 5: 1, 6: -1}
    }
    
    # We need to build the full vectors for each (k, l) pair.
    # It's easier to iterate k, then l.
    # For a fixed k, the U vector is sum over blocks b of S_coeffs[k][b] * (base_u[l] embedded in block b)
    # The V vector is sum over blocks b of T_coeffs[k][b] * (base_v[l] embedded in block b)
    # The W vector is sum over blocks c of C_coeffs[c][k] * (base_w[l] embedded in block c)
    
    for k in range(7):
        # Build U_16 for this k and all l
        U_16s = [[0]*16 for _ in range(7)]
        for l in range(7):
            for b, coeff in S_coeffs[k].items():
                idxs = A_blocks[b]
                for j, idx in enumerate(idxs):
                    U_16s[l][idx] += coeff * base_u[l][j]
            
        # Build V_24 for this k and all l
        V_24s = [[0]*24 for _ in range(7)]
        for l in range(7):
            for b, coeff in T_coeffs[k].items():
                idxs = B_blocks[b]
                for j, idx in enumerate(idxs):
                    V_24s[l][idx] += coeff * base_v[l][j]
        
        # Build W_24 for this k and all l
        W_24s = [[0]*24 for _ in range(7)]
        for l in range(7):
            for c, coeffs in C_coeffs.items():
                if k+1 in coeffs: # k is 0-indexed, coeffs keys are 1-indexed
                    coeff = coeffs[k+1]
                    idxs = C_blocks[c]
                    for j, idx in enumerate(idxs):
                        W_24s[l][idx] += coeff * base_w[l][j]
                        
        for l in range(7):
            add_term(U_16s[l], V_24s[l], W_24s[l])
            
    # --- Part 2: 4x4 by 4x2 (Last 2 columns of B) ---
    # B indices for last 2 columns: j in 0..3, k in 4..5
    # B_top: rows 0-1, cols 4-5 -> idx 4,5,10,11
    # B_bot: rows 2-3, cols 4-5 -> idx 16,17,22,23
    
    # C indices for last 2 cols: k in 4..5, i in 0..3
    # C_top: rows 4-5, cols 0-1 -> idx 4,5,8,9? No.
    # W index is k*n + i. n=4.
    # k=4, i=0 -> 16
    # k=4, i=1 -> 17
    # k=4, i=2 -> 18
    # k=4, i=3 -> 19
    # k=5, i=0 -> 20
    # k=5, i=1 -> 21
    # k=5, i=2 -> 22
    # k=5, i=3 -> 23
    
    # C_top (rows 4-5, cols 0-1):
    # k=4, i=0,1 -> 16,17
    # k=5, i=0,1 -> 20,21
    # Indices: 16,17,20,21
    
    # C_bot (rows 4-5, cols 2-3):
    # k=4, i=2,3 -> 18,19
    # k=5, i=2,3 -> 22,23
    # Indices: 18,19,22,23
    
    # A blocks same as before.
    # We need to compute:
    # N1 = A1 * B_top
    # N2 = A2 * B_bot
    # N3 = A3 * B_top
    # N4 = A4 * B_bot
    
    # C_top = N1 + N2
    # C_bot = N3 + N4
    
    # For each N (4 total), we use the 7 base tensors.
    # Total 28 terms.
    
    # B_top indices: 4,5,10,11
    # B_bot indices: 16,17,22,23
    
    # C_top indices: 16,17,20,21
    # C_bot indices: 18,19,22,23
    
    # N1: A1 * B_top -> C_top
    # N2: A2 * B_bot -> C_top
    # N3: A3 * B_top -> C_bot
    # N4: A4 * B_bot -> C_bot
    
    for l in range(7):
        # N1
        u_16 = [0]*16
        v_24 = [0]*24
        w_24 = [0]*24
        idxs_A = A_blocks[1]
        idxs_B = [4,5,10,11]
        idxs_C = [16,17,20,21]
        for j in range(4):
            u_16[idxs_A[j]] = base_u[l][j]
            v_24[idxs_B[j]] = base_v[l][j]
            w_24[idxs_C[j]] = base_w[l][j]
        add_term(u_16, v_24, w_24)
        
        # N2
        u_16 = [0]*16
        v_24 = [0]*24
        w_24 = [0]*24
        idxs_A = A_blocks[2]
        idxs_B = [16,17,22,23]
        idxs_C = [16,17,20,21]
        for j in range(4):
            u_16[idxs_A[j]] = base_u[l][j]
            v_24[idxs_B[j]] = base_v[l][j]
            w_24[idxs_C[j]] = base_w[l][j]
        add_term(u_16, v_24, w_24)
        
        # N3
        u_16 = [0]*16
        v_24 = [0]*24
        w_24 = [0]*24
        idxs_A = A_blocks[3]
        idxs_B = [4,5,10,11]
        idxs_C = [18,19,22,23]
        for j in range(4):
            u_16[idxs_A[j]] = base_u[l][j]
            v_24[idxs_B[j]] = base_v[l][j]
            w_24[idxs_C[j]] = base_w[l][j]
        add_term(u_16, v_24, w_24)
        
        # N4
        u_16 = [0]*16
        v_24 = [0]*24
        w_24 = [0]*24
        idxs_A = A_blocks[4]
        idxs_B = [16,17,22,23]
        idxs_C = [18,19,22,23]
        for j in range(4):
            u_16[idxs_A[j]] = base_u[l][j]
            v_24[idxs_B[j]] = base_v[l][j]
            w_24[idxs_C[j]] = base_w[l][j]
        add_term(u_16, v_24, w_24)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
