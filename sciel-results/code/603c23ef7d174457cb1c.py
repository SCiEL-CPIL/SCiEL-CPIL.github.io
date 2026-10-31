import time

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # We expect n=4, m=4, p=4
    if n != 4 or m != 4 or p != 4:
        # Fallback to schoolbook if dimensions don't match
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin

    # Strassen definitions for 2x2
    # Indices for 2x2 blocks:
    # Block (bi, bj) of 4x4 matrix consists of 2x2 scalars
    # Scalar indices in 4x4:
    # 0,1
    # 2,3
    # 4,5
    # 6,7
    # 8,9
    # 10,11
    # 12,13
    # 14,15
    # 
    # Block 00 (top-left): 0,1,2,3
    # Block 01 (top-right): 4,5,6,7
    # Block 10 (bottom-left): 8,9,10,11
    # Block 11 (bottom-right): 12,13,14,15

    # Strassen Level 1 (4x4 -> 7 products of 2x2)
    # M0 = (A00 + A11) * (B00 + B11)
    # M1 = (A10 + A11) * B00
    # M2 = A00 * (B01 - B11)
    # M3 = A11 * (B10 - B00)
    # M4 = (A00 + A01) * B11
    # M5 = (A10 - A00) * (B00 + B01)
    # M6 = (A01 - A11) * (B10 + B11)
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5

    # For each Mi, we define the linear combinations of blocks
    # Then for each Mi, we apply Strassen Level 2 (2x2 -> 7 products of 1x1)
    
    # Helper to get scalar indices for a block
    def get_block_indices(bi, bj):
        # bi, bj in {0, 1}
        # Returns list of 4 scalar indices (r0c0, r0c1, r1c0, r1c1)
        base = bi * 8 + bj * 2
        return [base, base+1, base+2, base+3]

    # Strassen Level 2 (2x2 -> 7 scalars)
    # S0 = (X00 + X11)(Y00 + Y11)
    # S1 = (X10 + X11)Y00
    # S2 = X00(Y01 - Y11)
    # S3 = X11(Y10 - Y00)
    # S4 = (X00 + X01)Y11
    # S5 = (X10 - X00)(Y00 + Y01)
    # S6 = (X01 - X11)(Y10 + Y11)
    
    # Z00 = S0 + S3 - S4 + S6
    # Z01 = S2 + S4
    # Z10 = S1 + S3
    # Z11 = S0 - S1 + S2 + S5

    # We need to map each scalar product to U, V, W vectors
    
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    col = 0
    
    # Define L1 factors
    # Each factor is a list of (block_idx, sign)
    L1_A = [
        [(0, 1), (3, 1)],  # M0: A00 + A11
        [(2, 1), (3, 1)],  # M1: A10 + A11
        [(0, 1)],          # M2: A00
        [(3, 1)],          # M3: A11
        [(0, 1), (1, 1)],  # M4: A00 + A01
        [(2, 1), (0, -1)], # M5: A10 - A00
        [(1, 1), (3, -1)]  # M6: A01 - A11
    ]
    L1_B = [
        [(0, 1), (3, 1)],  # M0: B00 + B11
        [(0, 1)],          # M1: B00
        [(1, 1), (3, -1)], # M2: B01 - B11
        [(2, 1), (0, -1)], # M3: B10 - B00
        [(3, 1)],          # M4: B11
        [(0, 1), (1, 1)],  # M5: B00 + B01
        [(2, 1), (3, 1)]   # M6: B10 + B11
    ]
    
    # L1 Coefficients for C blocks
    # C_block = sum coeff * Mi
    # C00: M0(1), M1(0), M2(0), M3(1), M4(-1), M5(0), M6(1)
    # C01: M0(0), M1(0), M2(1), M3(0), M4(1), M5(0), M6(0)
    # C10: M0(0), M1(1), M2(0), M3(1), M4(0), M5(0), M6(0)
    # C11: M0(1), M1(-1), M2(1), M3(0), M4(0), M5(1), M6(0)
    
    L1_C_Coeffs = [
        [1, 0, 0, 1, -1, 0, 1],  # C00 coeffs for M0..M6
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    # L2 Factors (for 2x2 Strassen)
    # X and Y are 2x2 matrices.
    # We need to expand X and Y into scalar combinations.
    # For a given L1 term, X is a sum of blocks.
    # Each block is 2x2.
    # So X is a 2x2 matrix where each entry is a linear combination of scalars.
    
    # We iterate L1 (0..6) and L2 (0..6)
    for l1 in range(7):
        # Construct X and Y linear combinations for this L1 term
        # X will be a list of 4 vectors (for X00, X01, X10, X11)
        # Each vector is a list of (scalar_idx, sign)
        X = [[], [], [], []]
        Y = [[], [], [], []]
        
        for b_idx, sign in L1_A[l1]:
            inds = get_block_indices(b_idx // 2, b_idx % 2) # Wait, block index 0..3
            # b_idx 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            # get_block_indices takes bi, bj
            bi, bj = b_idx // 2, b_idx % 2
            inds = get_block_indices(bi, bj)
            # Map block entries to X positions
            # Block is 2x2.
            # X[0] (X00) gets Block[0]
            # X[1] (X01) gets Block[1]
            # X[2] (X10) gets Block[2]
            # X[3] (X11) gets Block[3]
            for p in range(4):
                X[p].append((inds[p], sign))
                
        for b_idx, sign in L1_B[l1]:
            bi, bj = b_idx // 2, b_idx % 2
            inds = get_block_indices(bi, bj)
            for p in range(4):
                Y[p].append((inds[p], sign))

        # Now apply L2 Strassen
        # S0 = (X00 + X11)(Y00 + Y11)
        # S1 = (X10 + X11)Y00
        # S2 = X00(Y01 - Y11)
        # S3 = X11(Y10 - Y00)
        # S4 = (X00 + X01)Y11
        # S5 = (X10 - X00)(Y00 + Y01)
        # S6 = (X01 - X11)(Y10 + Y11)
        
        L2_A = [
            [X[0], X[3]], # S0 A
            [X[2], X[3]], # S1 A
            [X[0]],       # S2 A
            [X[3]],       # S3 A
            [X[0], X[1]], # S4 A
            [X[2], X[0]], # S5 A (X10 - X00)
            [X[1], X[3]]  # S6 A (X01 - X11)
        ]
        # Signs for L2 A
        L2_A_Signs = [
            [1, 1],
            [1, 1],
            [1],
            [1],
            [1, 1],
            [1, -1],
            [1, -1]
        ]
        
        L2_B = [
            [Y[0], Y[3]], # S0 B
            [Y[0]],       # S1 B
            [Y[1], Y[3]], # S2 B (Y01 - Y11)
            [Y[2], Y[0]], # S3 B (Y10 - Y00)
            [Y[3]],       # S4 B
            [Y[0], Y[1]], # S5 B (Y00 + Y01)
            [Y[2], Y[3]]  # S6 B (Y10 + Y11)
        ]
        L2_B_Signs = [
            [1, 1],
            [1],
            [1, -1],
            [1, -1],
            [1],
            [1, 1],
            [1, 1]
        ]
        
        # L2 C Coeffs (Z reconstruction)
        # Z00 = S0 + S3 - S4 + S6
        # Z01 = S2 + S4
        # Z10 = S1 + S3
        # Z11 = S0 - S1 + S2 + S5
        L2_Z_Coeffs = [
            [1, 0, 0, 1, -1, 0, 1],  # Z00
            [0, 0, 1, 0, 1, 0, 0],   # Z01
            [0, 1, 0, 1, 0, 0, 0],   # Z10
            [1, -1, 1, 0, 0, 1, 0]   # Z11
        ]
        
        for l2 in range(7):
            if time.time() > deadline:
                break
                
            # Construct U and V vectors for this scalar product
            u_vec = [0] * 16
            v_vec = [0] * 16
            
            # A side
            for i_part in range(len(L2_A[l2])):
                sign = L2_A_Signs[l2][i_part]
                for idx, s in L2_A[l2][i_part]:
                    u_vec[idx] += sign * s
                    
            # B side
            for i_part in range(len(L2_B[l2])):
                sign = L2_B_Signs[l2][i_part]
                for idx, s in L2_B[l2][i_part]:
                    v_vec[idx] += sign * s
            
            # W vector: contributions to C
            # This scalar product (S_l2) contributes to Z_u, v with coeff L2_Z_Coeffs[u, v, l2]
            # Z_u, v is the (u,v) entry of the 2x2 result of M_l1
            # M_l1 contributes to C_block with coeff L1_C_Coeffs[block, l1]
            
            w_vec = [0] * 16
            
            # Map Z_u, v to C indices
            # Z is 2x2. u in {0,1}, v in {0,1}
            # M_l1 is a block.
            # Which block of C does M_l1 contribute to?
            # M0 -> C00
            # M1 -> C10
            # M2 -> C01
            # M3 -> C10 (and C00? No, see L1_C_Coeffs)
            # Actually, L1_C_Coeffs defines which M contributes to which C_block.
            # C_block is 0: C00, 1: C01, 2: C10, 3: C11
            
            for c_block in range(4):
                coeff_c = L1_C_Coeffs[c_block][l1]
                if coeff_c == 0:
                    continue
                
                for u in range(2):
                    for v in range(2):
                        coeff_z = L2_Z_Coeffs[u][v][l2] # Note: L2_Z_Coeffs[u][v] is list of 7?
                        # My L2_Z_Coeffs definition:
                        # L2_Z_Coeffs = [
                        #   [1, 0, 0, 1, -1, 0, 1],  # Z00 coeffs for S0..S6
                        #   ...
                        # ]
                        # So L2_Z_Coeffs[u][v] is not correct.
                        # L2_Z_Coeffs is a list of 4 lists (for Z00, Z01, Z10, Z11)
                        # Each list has 7 entries (coeff for S0..S6)
                        # So coeff_z = L2_Z_Coeffs[c_idx][l2] where c_idx = u*2 + v?
                        # Let's re-verify indices.
                        # Z00 is u=0, v=0. Index 0.
                        # Z01 is u=0, v=1. Index 1.
                        # Z10 is u=1, v=0. Index 2.
                        # Z11 is u=1, v=1. Index 3.
                        z_idx = u * 2 + v
                        coeff_z = L2_Z_Coeffs[z_idx][l2]
                        
                        if coeff_z == 0:
                            continue
                        
                        total_coeff = coeff_c * coeff_z
                        
                        # Determine C index
                        # c_block 0: C00 (Top-Left 2x2)
                        # c_block 1: C01 (Top-Right 2x2)
                        # c_block 2: C10 (Bottom-Left 2x2)
                        # c_block 3: C11 (Bottom-Right 2x2)
                        
                        bi, bj = c_block // 2, c_block % 2
                        # C index for (bi, bj) block, entry (u, v)
                        # Row I = bi * 2 + u
                        # Col J = bj * 2 + v
                        I = bi * 2 + u
                        J = bj * 2 + v
                        
                        # W index is J * n + I (column-major)
                        w_idx = J * 4 + I
                        w_vec[w_idx] += total_coeff
            
            U[:, col] = u_vec
            V[:, col] = v_vec
            W[:, col] = w_vec
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
