"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a rank-28 decomposition for 4x4x4 (Strassen top, Schoolbook bottom)
    # and apply it twice: once for the first 4 columns of B, once for the last 3 (padded to 4).
    # Total rank = 28 + 28 = 56.
    
    def get_4x4x4_decomp():
        # Returns U, V, W for 4x4x4 with rank 28
        rank = 28
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        
        # Strassen terms for 4x4
        # P_k and Q_k are 2x2 blocks
        # k: 0..6
        # Each term uses 4 schoolbook multiplications for the 2x2 blocks
        
        # Define block indices for A and B
        # A blocks: (0,0), (0,1), (1,0), (1,1)
        # Local indices i, j in 0..1
        # Global i = I*2 + i
        
        # Strassen Formulas:
        # M0 = (A00+A11)(B00+B11)
        # M1 = (A10+A11)B00
        # M2 = A00(B01-B11)
        # M3 = A11(B10-B00)
        # M4 = (A00+A01)B11
        # M5 = (A10-A00)(B00+B01)
        # M6 = (A01-A11)(B10+B11)
        
        # Reconstruction:
        # C00 = M0 + M2 - M3 + M6
        # C01 = M4 + M5
        # C10 = M1 + M3
        # C11 = M0 - M2 + M4 + M5
        
        # We will add terms for each scalar multiplication in M_k = P_k Q_k
        # Schoolbook for 2x2:
        # t0 = P00 Q00
        # t1 = P01 Q10
        # t2 = P10 Q01
        # t3 = P11 Q11
        
        # For each k, we define the coefficients of A blocks in P_k and B blocks in Q_k
        # P_k = sum c_ik A_ik
        # Q_k = sum c_jk B_jk
        
        terms = []
        
        # Helper to add a schoolbook term
        # P_block_coeffs: dict (I,J) -> coeff
        # Q_block_coeffs: dict (I,J) -> coeff
        # We need to expand P_k Q_k into scalar mults
        # P_k is 2x2. Q_k is 2x2.
        # P_k[i][l] = sum_{I,J} c_{I,J}^P A_{I,J}[i-2I][l-2J] ? No.
        # A is 4x4. A_{I,J} is 2x2 block at (2I, 2J).
        # P_k is a 2x2 matrix formed by adding blocks.
        # So P_k[u][v] = sum_{I,J} c_{I,J} A[2I+u][2J+v]
        
        # We iterate over the 4 scalar mults of the 2x2 product
        # u in 0..1, v in 0..1 for P and Q?
        # M_k[u][v] = sum_{l=0}^1 P_k[u][l] Q_k[l][v]
        
        # We add a term for each (u, l, v) ? No, schoolbook has 4 mults:
        # (u=0, l=0, v=0) -> P[0][0] Q[0][0]
        # (u=0, l=1, v=1) -> P[0][1] Q[1][1] ? No.
        # Schoolbook:
        # M[0][0] = P[0][0]Q[0][0] + P[0][1]Q[1][0]
        # M[0][1] = P[0][0]Q[0][1] + P[0][1]Q[1][1]
        # M[1][0] = P[1][0]Q[0][0] + P[1][1]Q[1][0]
        # M[1][1] = P[1][0]Q[0][1] + P[1][1]Q[1][1]
        
        # The 4 independent multiplications are:
        # 1. P[0][0] * Q[0][0]
        # 2. P[0][1] * Q[1][0]
        # 3. P[1][0] * Q[0][1]
        # 4. P[1][1] * Q[1][1]
        
        # For each of these, we need to map to global indices.
        
        def add_term(U, V, W, col_idx, p_coeffs, q_coeffs, u, l, v, w_coeff):
            # p_coeffs: dict (I,J) -> coeff for P[u][l]
            # q_coeffs: dict (I,J) -> coeff for Q[l][v]
            # w_coeff: coefficient for M_k[u][v] in final C block
            
            # Actually, we need to sum the contributions to C.
            # C_block[I_C][J_C] gets contribution from M_k[u][v]
            
            # This is getting complex to implement generically inside the loop.
            # Let's just hardcode the 7x4 = 28 terms.
            pass
            
        # Let's define the 7 Strassen steps and their 4 sub-multiplications.
        # We will build U, V, W directly.
        
        # Step 0: M0 = (A00+A11)(B00+B11)
        # P0 = A00 + A11. Q0 = B00 + B11.
        # Sub-mults:
        # 1. P[0][0] Q[0][0] = (A00[0][0]+A11[0][0]) * (B00[0][0]+B11[0][0])
        #    This expands to 4 scalar mults if we don't combine?
        #    No, the tensor decomposition allows linear combinations in U and V.
        #    U[i,j] can be 1, -1, 0.
        #    So for term "A00[0][0] * B00[0][0]", we have:
        #    U[0*4+0, c] = 1
        #    V[0*4+0, c] = 1
        #    W[0*4+0, c] = 1 (contributes to C00)
        
        # We will generate the 28 terms by iterating k=0..6, sub=0..3.
        
        # Define Strassen P, Q block coefficients
        # P_k is sum of A blocks. Q_k is sum of B blocks.
        # A blocks: (0,0), (0,1), (1,0), (1,1)
        
        strassen_defs = [
            # (P_coeffs, Q_coeffs)
            # P_coeffs: list of (I, J, coeff)
            # Q_coeffs: list of (I, J, coeff)
            [
                ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]), # M0
                ([(1,0,1), (1,1,1)], [(0,0,1)]),           # M1
                ([(0,0,1)], [(0,1,1), (1,1,-1)]),          # M2
                ([(1,1,1)], [(1,0,1), (0,0,-1)]),          # M3
                ([(0,0,1), (0,1,1)], [(1,1,1)]),           # M4
                ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]), # M5
                ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]), # M6
            ]
        ]
        
        # Reconstruction Coeffs for C blocks
        # C00: M0, M2, -M3, M6
        # C01: M4, M5
        # C10: M1, M3
        # C11: -M0, M2, M4, M5  (Wait, C11 = M0 - M2 + M4 + M5? Let's check)
        # Standard:
        # C11 = M1 + M3 - M4 + M7 (1-based) -> M0 + M2 - M3 + M6 (0-based)
        # C12 = M5 + M6 -> M4 + M5
        # C21 = M2 + M4 -> M1 + M3
        # C22 = M1 - M3 + M5 + M6 -> M0 - M2 + M4 + M5
        
        recon = {
            (0,0): [(0,1), (2,1), (3,-1), (6,1)],
            (0,1): [(4,1), (5,1)],
            (1,0): [(1,1), (3,1)],
            (1,1): [(0,1), (2,-1), (4,1), (5,1)]
        }
        
        col = 0
        for k in range(7):
            p_coeffs, q_coeffs = strassen_defs[0][k]
            # 4 schoolbook sub-multiplications
            # (u, l, v) pairs for P[u][l] * Q[l][v]
            # 1. u=0, l=0, v=0
            # 2. u=0, l=1, v=1 -> No, P[0][1] * Q[1][0] is u=0, l=1, v=0?
            # M[0][0] = P[0][0]Q[0][0] + P[0][1]Q[1][0]
            # M[0][1] = P[0][0]Q[0][1] + P[0][1]Q[1][1]
            # M[1][0] = P[1][0]Q[0][0] + P[1][1]Q[1][0]
            # M[1][1] = P[1][0]Q[0][1] + P[1][1]Q[1][1]
            
            # The 4 mults are:
            # 1. P[0][0] * Q[0][0]
            # 2. P[0][1] * Q[1][0]
            # 3. P[1][0] * Q[0][1]
            # 4. P[1][1] * Q[1][1]
            
            sub_mults = [
                (0, 0, 0), # u, l, v
                (0, 1, 0), # u, l, v -> P[0][1] Q[1][0]
                (1, 0, 1), # u, l, v -> P[1][0] Q[0][1]
                (1, 1, 1)  # u, l, v -> P[1][1] Q[1][1]
            ]
            
            for u, l, v in sub_mults:
                # This sub-multiplication contributes to M_k[u][v]
                # We need to add terms for each (I,J) in p_coeffs and (I',J') in q_coeffs
                # P[u][l] = sum_{I,J} c_p A[2I+u][2J+l]
                # Q[l][v] = sum_{I',J'} c_q B[2I'+l][2J'+v]
                
                # The product is sum c_p c_q A[...] B[...]
                # Each A[...] B[...] is a scalar multiplication.
                # We can add a term for each pair.
                
                # However, this would increase the rank!
                # Rank is the number of columns in U,V,W.
                # If we expand the sums, we get more terms.
                # We must keep the linear combinations in U and V.
                
                # So, for this specific sub-multiplication (u,l,v) of M_k:
                # U part: For each (I,J,c_p) in p_coeffs:
                #   U[ (2I+u)*4 + (2J+l) , col] += c_p
                # V part: For each (I',J',c_q) in q_coeffs:
                #   V[ (2I'+l)*4 + (2J'+v) , col] += c_q
                
                # W part:
                # M_k[u][v] contributes to C blocks.
                # For each (I_C, J_C, w_c) in recon[(I_C, J_C)]:
                #   If k == I_C? No, recon maps M_k to C blocks.
                #   The contribution to C[I_C*2+u][J_C*2+v] is w_c * M_k[u][v].
                #   So W[ (J_C*2+v)*4 + (I_C*2+u) , col] += w_c
                
                # Wait, the W index is k*n + i.
                # k is col of C. i is row of C.
                # C row = I_C*2 + u.
                # C col = J_C*2 + v.
                
                # So:
                for I, J, c_p in p_coeffs:
                    row_a = 2*I + u
                    col_a = 2*J + l
                    U[row_a * 4 + col_a][col] += c_p
                
                for I, J, c_q in q_coeffs:
                    row_b = 2*I + l
                    col_b = 2*J + v
                    V[row_b * 4 + col_b][col] += c_q
                
                # W updates
                for I_C, J_C, w_c in recon[(0,0)] + recon[(0,1)] + recon[(1,0)] + recon[(1,1)]:
                    # Check if this M_k (index k) is in the list
                    # recon values are lists of (k, coeff)
                    pass # Need to check if k matches
                    
                # Correct W update:
                for I_C in range(2):
                    for J_C in range(2):
                        for k_idx, coeff in recon[(I_C, J_C)]:
                            if k_idx == k:
                                row_c = 2*I_C + u
                                col_c = 2*J_C + v
                                W[col_c * 4 + row_c][col] += coeff
                                
                col += 1
        
        return U, V, W

    U1, V1, W1 = get_4x4x4_decomp()
    U2, V2, W2 = get_4x4x4_decomp()
    
    total_rank = 56
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(28)]
    W = [[0] * total_rank for _ in range(28)]
    
    # Copy 1: Cols 0-3 of B
    for i in range(16):
        for c in range(28):
            U[i][c] = U1[i][c]
    
    for i in range(16): # j*4+k
        j = i // 4
        k = i % 4
        if k < 4:
            for c in range(28):
                V[j * 7 + k][c] = V1[i][c]
                
    for i in range(16): # k*4+i
        k = i // 4
        i_row = i % 4
        if k < 4:
            for c in range(28):
                W[k * 4 + i_row][c] = W1[i][c]
                
    # Copy 2: Cols 4-6 of B (padded to 4)
    offset_c = 28
    for i in range(16):
        for c in range(28):
            U[i][c + offset_c] = U2[i][c]
            
    for i in range(16): # j*4+k
        j = i // 4
        k = i % 4
        if k < 3: # Only first 3 cols are real
            for c in range(28):
                V[j * 7 + (k + 4)][c + offset_c] = V2[i][c]
        # k=3 is padding, V remains 0
                
    for i in range(16): # k*4+i
        k = i // 4
        i_row = i % 4
        if k < 3:
            for c in range(28):
                W[(k + 4) * 4 + i_row][c + offset_c] = W2[i][c]
        # k=3 is padding, W remains 0

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
