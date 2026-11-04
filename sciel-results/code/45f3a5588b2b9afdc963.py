# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    DEADLINE = 145
    
    n, m, p = 4, 4, 5
    
    # Rank 65 decomposition:
    # Part 1: 4x4x4 multiplication using 2-level Strassen (Rank 49)
    # Part 2: 4x4x1 multiplication using Schoolbook (Rank 16)
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs, r_idx):
        for i in range(n):
            for j in range(m):
                U[i * m + j][r_idx] += u_coeffs[i * m + j]
        for j in range(m):
            for k in range(p):
                V[j * p + k][r_idx] += v_coeffs[j * p + k]
        for k in range(p):
            for i in range(n):
                W[k * n + i][r_idx] += w_coeffs[k * n + i]

    # --- Part 1: 4x4x4 (Cols 0-3 of B, Rows 0-3 of C) ---
    
    # Outer Strassen Coefficients
    # X_m = linear combo of A blocks
    # Y_m = linear combo of B blocks
    # A blocks: A00, A01, A10, A11 (2x2 each)
    # B blocks: B00, B01, B10, B11 (2x2 each, B is 4x4 part)
    
    # X definitions (coeffs for A00, A01, A10, A11)
    X_defs = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1]  # M7: A01 - A11
    ]
    
    # Y definitions (coeffs for B00, B01, B10, B11)
    Y_defs = [
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [0, 0, 1, -1], # M4: B10 - B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1]   # M7: B10 + B11
    ]
    
    # C block contributions from M_m
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # C blocks: C00, C01, C10, C11 (2x2 each)
    # C00 is rows 0-1, cols 0-1
    # C01 is rows 0-1, cols 2-3
    # C10 is rows 2-3, cols 0-1
    # C11 is rows 2-3, cols 2-3
    
    C_contribs = [
        [1, 0, 0, 1],   # M1 -> C00(1), C11(1)
        [0, 0, 1, -1],  # M2 -> C10(1), C11(-1)
        [0, 1, 0, 1],   # M3 -> C01(1), C11(1)
        [1, 0, 1, 0],   # M4 -> C00(1), C10(1)
        [-1, 1, 0, 0],  # M5 -> C00(-1), C01(1)
        [0, 0, 0, 1],   # M6 -> C11(1)
        [1, 0, 0, 0]    # M7 -> C00(1)
    ]
    
    # Inner Strassen Coefficients
    # For Z = L R (2x2 matrices)
    # P1 = (L00+L11)(R00+R11)
    # P2 = (L10+L11)R00
    # P3 = L00(R01-R11)
    # P4 = L11(R10-R00)
    # P5 = (L00+L01)R11
    # P6 = (L10-L00)(R00+R01)
    # P7 = (L01-L11)(R10+R11)
    
    # L coeffs (for L00, L01, L10, L11)
    L_inner = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # R coeffs (for R00, R01, R10, R11)
    R_inner = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # Z block contributions from P_s
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    Z_contribs = [
        [1, 0, 0, 1],   # P1 -> Z00(1), Z11(1)
        [0, 0, 1, -1],  # P2 -> Z10(1), Z11(-1)
        [0, 1, 0, 1],   # P3 -> Z01(1), Z11(1)
        [1, 0, 1, 0],   # P4 -> Z00(1), Z10(1)
        [-1, 1, 0, 0],  # P5 -> Z00(-1), Z01(1)
        [0, 0, 0, 1],   # P6 -> Z11(1)
        [1, 0, 0, 0]    # P7 -> Z00(1)
    ]
    
    r_idx = 0
    for m_idx in range(7):
        # X_m is formed by X_defs[m_idx]
        # Y_m is formed by Y_defs[m_idx]
        
        for s_idx in range(7):
            # L_s is formed by L_inner[s_idx] from X_m blocks
            # R_s is formed by R_inner[s_idx] from Y_m blocks
            
            # Calculate U coeffs (linear combo of A)
            u_coeffs = [0] * (n * m)
            # X_m blocks are 2x2 matrices of A
            # X_m[u,v] = sum_{b_idx} X_defs[m_idx][b_idx] * A_block[b_idx]
            # L_s[a,b] = sum_{u,v} L_inner[s_idx][u*2+v] * X_m[u,v]
            # So L_s[a,b] = sum_{u,v,b_idx} L_inner[s_idx][u*2+v] * X_defs[m_idx][b_idx] * A_block[b_idx][a,b]
            
            # Map block index to A indices
            # Block 0: A00 (rows 0-1, cols 0-1)
            # Block 1: A01 (rows 0-1, cols 2-3)
            # Block 2: A10 (rows 2-3, cols 0-1)
            # Block 3: A11 (rows 2-3, cols 2-3)
            
            for b_idx in range(4):
                coeff_b = X_defs[m_idx][b_idx]
                if coeff_b == 0: continue
                
                # Determine row/col offset for this block
                block_row = (b_idx // 2) * 2
                block_col = (b_idx % 2) * 2
                
                for u in range(2):
                    for v in range(2):
                        # L_inner index for block (u,v)
                        # Wait, L_inner[s_idx] is coeffs for L00, L01, L10, L11
                        # L00 corresponds to X_m[0,0] which is Block 0?
                        # No, X_m is a 2x2 matrix of blocks.
                        # X_m[0,0] is Block 0 (A00 part).
                        # X_m[0,1] is Block 1 (A01 part).
                        # X_m[1,0] is Block 2 (A10 part).
                        # X_m[1,1] is Block 3 (A11 part).
                        
                        # L_inner[s_idx] indices:
                        # 0: L00 (from X_m[0,0])
                        # 1: L01 (from X_m[0,1])
                        # 2: L10 (from X_m[1,0])
                        # 3: L11 (from X_m[1,1])
                        
                        # So for block b_idx, the corresponding L_inner index is b_idx.
                        # Wait, b_idx 0 is X_m[0,0]. b_idx 1 is X_m[0,1].
                        # Yes.
                        
                        l_coeff = L_inner[s_idx][b_idx]
                        if l_coeff == 0: continue
                        
                        total_coeff = coeff_b * l_coeff
                        if total_coeff == 0: continue
                        
                        # Add to U for all entries in this block
                        for a in range(2):
                            for c in range(2):
                                i = block_row + a
                                j = block_col + c
                                u_coeffs[i * m + j] += total_coeff
            
            # Calculate V coeffs (linear combo of B)
            v_coeffs = [0] * (m * p)
            for b_idx in range(4):
                coeff_b = Y_defs[m_idx][b_idx]
                if coeff_b == 0: continue
                
                block_row = (b_idx // 2) * 2
                block_col = (b_idx % 2) * 2
                
                for u in range(2):
                    for v in range(2):
                        r_coeff = R_inner[s_idx][b_idx]
                        if r_coeff == 0: continue
                        
                        total_coeff = coeff_b * r_coeff
                        if total_coeff == 0: continue
                        
                        for a in range(2):
                            for c in range(2):
                                j = block_row + a
                                k = block_col + c
                                # B is 4x4 for this part (cols 0-3)
                                v_coeffs[j * p + k] += total_coeff
            
            # Calculate W coeffs (linear combo of C)
            w_coeffs = [0] * (p * n)
            # C_block contributions
            # Z_contribs[s_idx] gives coeffs for Z00, Z01, Z10, Z11
            # C_contribs[m_idx] gives coeffs for C00, C01, C10, C11
            # Total coeff for C_block b_c is sum_{b_z} Z_contribs[s_idx][b_z] * C_contribs[m_idx][b_c]
            # Wait, Z is M_m.
            # M_m[u,v] is a 2x2 matrix.
            # Z_contribs[s_idx][b_z] is the coeff of P_s in M_m[b_z_row, b_z_col].
            # b_z is 0..3 corresponding to M_m[0,0], M_m[0,1], M_m[1,0], M_m[1,1].
            
            for b_z in range(4):
                z_coeff = Z_contribs[s_idx][b_z]
                if z_coeff == 0: continue
                
                # M_m[b_z] contributes to C blocks
                # C_contribs[m_idx] is coeffs for C00, C01, C10, C11
                # M_m[b_z] is a block of M_m.
                # Does M_m[b_z] go to C[b_z]?
                # Yes, if the block structures align.
                # M_m is 2x2 blocks. C is 2x2 blocks.
                # M_m[0,0] -> C00.
                # M_m[0,1] -> C01.
                # M_m[1,0] -> C10.
                # M_m[1,1] -> C11.
                
                c_coeff = C_contribs[m_idx][b_z]
                if c_coeff == 0: continue
                
                total_coeff = z_coeff * c_coeff
                if total_coeff == 0: continue
                
                # Add to W for all entries in C block b_z
                block_row = (b_z // 2) * 2
                block_col = (b_z % 2) * 2
                
                for a in range(2):
                    for c in range(2):
                        i = block_row + a
                        k = block_col + c
                        # W index: k * n + i
                        w_coeffs[k * n + i] += total_coeff
            
            add_term(u_coeffs, v_coeffs, w_coeffs, r_idx)
            r_idx += 1
            
            if time.time() - start_time > DEADLINE - 5:
                # Fallback to schoolbook if too slow (should not happen)
                return {"U": [[0]*80 for _ in range(16)], "V": [[0]*80 for _ in range(20)], "W": [[0]*80 for _ in range(20)]}

    # --- Part 2: 4x4x1 (Col 4 of B, Rows 0-3 of C) ---
    # Schoolbook: C[i,4] = sum_j A[i,j] B[j,4]
    for i in range(4):
        for j in range(4):
            u_coeffs = [0] * (n * m)
            v_coeffs = [0] * (m * p)
            w_coeffs = [0] * (p * n)
            
            u_coeffs[i * m + j] = 1
            v_coeffs[j * p + 4] = 1
            w_coeffs[4 * n + i] = 1
            
            add_term(u_coeffs, v_coeffs, w_coeffs, r_idx)
            r_idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
