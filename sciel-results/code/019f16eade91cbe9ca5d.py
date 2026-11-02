import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 block multiplications
    # Each is a tuple (A_coeff, B_coeff) where A_coeff is a 2x2 matrix of coefficients for A blocks
    # and B_coeff is a 2x2 matrix of coefficients for B blocks.
    # Blocks: 00, 01, 10, 11 correspond to 11, 12, 21, 22 in 1-based notation.
    
    # X0: (A00 + A11)(B00 + B11)
    A0 = [[1, 0], [0, 1]]
    B0 = [[1, 0], [0, 1]]
    
    # X1: (A10 + A11)B00
    A1 = [[0, 0], [1, 1]]
    B1 = [[1, 0], [0, 0]]
    
    # X2: A00(B01 - B11)
    A2 = [[1, 0], [0, 0]]
    B2 = [[0, 1], [0, -1]]
    
    # X3: A11(B10 - B00)
    A3 = [[0, 0], [0, 1]]
    B3 = [[-1, 0], [1, 0]]
    
    # X4: (A00 + A01)B11
    A4 = [[1, 1], [0, 0]]
    B4 = [[0, 0], [0, 1]]
    
    # X5: (A10 - A00)(B00 + B01)
    A5 = [[-1, 0], [1, 0]]
    B5 = [[1, 1], [0, 0]]
    
    # X6: (A01 - A11)(B10 + B11)
    A6 = [[0, 1], [-1, 0]]
    B6 = [[0, 0], [1, 1]]
    
    A_blocks = [A0, A1, A2, A3, A4, A5, A6]
    B_blocks = [B0, B1, B2, B3, B4, B5, B6]
    
    # Coefficients of Xk in C blocks
    # C00 = X0 + X3 - X4 + X6
    # C01 = X2 + X4
    # C10 = X1 + X3
    # C11 = X0 - X1 + X2 + X5
    K = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0]    # C11
    ]
    
    # Inner Strassen steps for 2x2 multiplication
    # Each step j defines linear forms for P (input A block) and Q (input B block)
    # P and Q are 2x2.
    # Step 0: (P00+P11)(Q00+Q11)
    # Step 1: (P10+P11)Q00
    # Step 2: P00(Q01-Q11)
    # Step 3: P11(Q10-Q00)
    # Step 4: (P00+P01)Q11
    # Step 5: (P10-P00)(Q00+Q01)
    # Step 6: (P01-P11)(Q10+Q11)
    
    # Define P coefficients for each step (2x2)
    P_steps = [
        [[1, 0], [0, 1]],  # 0
        [[0, 0], [1, 1]],  # 1
        [[1, 0], [0, 0]],  # 2
        [[0, 0], [0, 1]],  # 3
        [[1, 1], [0, 0]],  # 4
        [[-1, 0], [1, 0]], # 5
        [[0, 1], [-1, 0]]  # 6
    ]
    
    # Define Q coefficients for each step (2x2)
    Q_steps = [
        [[1, 0], [0, 1]],  # 0
        [[1, 0], [0, 0]],  # 1
        [[0, 1], [0, -1]], # 2
        [[-1, 0], [1, 0]], # 3
        [[0, 0], [0, 1]],  # 4
        [[1, 1], [0, 0]],  # 5
        [[0, 0], [1, 1]]   # 6
    ]
    
    # Coefficients of inner products m_j in the result R (2x2)
    # R00 = m0 + m3 - m4 + m6
    # R01 = m2 + m4
    # R10 = m1 + m3
    # R11 = m0 - m1 + m2 + m5
    R_coeff = [
        [1, 0, 0, 1, -1, 0, 1],  # R00
        [0, 0, 1, 0, 1, 0, 0],   # R01
        [0, 1, 0, 1, 0, 0, 0],   # R10
        [1, -1, 1, 0, 0, 1, 0]   # R11
    ]
    
    col = 0
    for k in range(7):
        for j in range(7):
            # Compute U and V coefficients
            # U is sum over i,j of u_ij A_ij
            # V is sum over j,k of v_jk B_jk
            
            # The input to the inner Strassen is P and Q.
            # P is a linear combination of A blocks.
            # P_ab = sum_{br, bc} A_blocks[k][br][bc] * A_block[br][bc]
            # Wait, A_blocks[k] is the coefficient matrix for the block multiplication k.
            # Let's call it AK. AK[br][bc] is the coefficient of A_block[br][bc] in the block P.
            # Actually, P is a 2x2 matrix.
            # P[0][0] = AK[0][0]*A00 + AK[0][1]*A01 + AK[1][0]*A10 + AK[1][1]*A11 ?
            # No, the block multiplication is defined on the 2x2 block matrices.
            # Let A_mat be the 2x2 matrix of blocks.
            # P = A_mat * A_blocks[k]? No.
            # In Strassen, P and Q are the inputs to the 7 multiplications.
            # For block k, the inputs to the 7 multiplications are linear combinations of the 4 blocks of A and B.
            # My A_blocks[k] definition above was slightly ambiguous.
            # Let's re-verify.
            # X0 = (A00 + A11)(B00 + B11).
            # Here P = [[A00, 0], [0, A11]]? No.
            # P is the matrix [[A00+A11, 0], [0, 0]]? No.
            # The formula is (A00 + A11) * (B00 + B11).
            # This is a product of two 2x2 matrices? No, A00 and A11 are 2x2 matrices.
            # So (A00 + A11) is a 2x2 matrix.
            # So P = A00 + A11. Q = B00 + B11.
            # So P is a 2x2 matrix.
            # P[0][0] = A00[0][0] + A11[0][0].
            # P[0][1] = A00[0][1] + A11[0][1].
            # P[1][0] = A00[1][0] + A11[1][0].
            # P[1][1] = A00[1][1] + A11[1][1].
            
            # So for each block k, we define P and Q as 2x2 matrices of linear forms.
            # Let's define the 4 blocks of A: A00, A01, A10, A11.
            # And 4 blocks of B: B00, B01, B10, B11.
            
            # I will explicitly define the 2x2 matrix P_k and Q_k for each k.
            
            # k=0: P = A00+A11, Q = B00+B11
            # k=1: P = A10+A11, Q = B00
            # k=2: P = A00, Q = B01-B11
            # k=3: P = A11, Q = B10-B00
            # k=4: P = A00+A01, Q = B11
            # k=5: P = A10-A00, Q = B00+B01
            # k=6: P = A01-A11, Q = B10+B11
            
            # Let's store these as 2x2 arrays of block indices and coefficients.
            # block_idx: 0=A00, 1=A01, 2=A10, 3=A11.
            
            P_defs = [
                [[(0, 1), (3, 1)], [(0, 0), (3, 0)], [(0, 0), (3, 0)], [(0, 0), (3, 0)]], # This is wrong structure.
                # Let's use a list of (block_idx, coeff) for each element of the 2x2 P matrix.
            ]
            
            # Correct P_defs:
            # P is 2x2.
            # P[0][0], P[0][1], P[1][0], P[1][1].
            
            P_defs = [
                # k=0
                [
                    [(0, 1), (3, 1)],  # P00 = A00 + A11
                    [(0, 0), (3, 0)],  # P01 = 0? No, A00+A11 is the whole matrix.
                    # Wait, A00 and A11 are 2x2 matrices.
                    # So P = A00 + A11 means P[0][0] = A00[0][0]+A11[0][0], etc.
                    # So for each element P[r][c], the linear form is sum of A00[r][c] and A11[r][c].
                ],
            ]
            
            # It is easier to just hardcode the linear forms for U and V for each of the 49 terms.
            # But I can generate them.
            
            # Let's define the block contributions for P and Q for each k.
            # P_blocks[k] is a 2x2 matrix where each entry is a dict {block_idx: coeff}
            
            P_blocks = [
                # k=0
                [
                    {0: 1, 3: 1}, {0: 0, 3: 0}, # P00, P01 ? No.
                    # P = A00 + A11.
                    # P[0][0] = A00[0][0] + A11[0][0].
                    # P[0][1] = A00[0][1] + A11[0][1].
                    # So for P[0][0], coeffs are {0:1, 3:1}.
                    # For P[0][1], coeffs are {0:1, 3:1}.
                    # For P[1][0], coeffs are {0:1, 3:1}.
                    # For P[1][1], coeffs are {0:1, 3:1}.
                ],
            ]
            
            # Let's just write a helper to expand.
            
            def expand_P(k, r, c):
                # Returns dict {global_idx: coeff} for A[r][c] part of P_k
                # P_k is defined by the 7 cases.
                # r, c are 0 or 1.
                if k == 0: # A00 + A11
                    return {0*4+0: 1, 0*4+1: 0, 1*4+0: 0, 1*4+1: 1} # No, global indices.
                    # A00 is rows 0,1 cols 0,1.
                    # A11 is rows 2,3 cols 2,3.
                    # P[r][c] = A00[r][c] + A11[r][c].
                    # A00[r][c] is A[r][c].
                    # A11[r][c] is A[r+2][c+2].
                    return {r*4+c: 1, (r+2)*4+(c+2): 1}
                elif k == 1: # A10 + A11
                    # A10 is rows 2,3 cols 0,1. A11 is rows 2,3 cols 2,3.
                    # P[r][c] = A10[r][c] + A11[r][c].
                    # A10[r][c] is A[r+2][c].
                    # A11[r][c] is A[r+2][c+2].
                    return {(r+2)*4+c: 1, (r+2)*4+(c+2): 1}
                elif k == 2: # A00
                    return {r*4+c: 1}
                elif k == 3: # A11
                    return {(r+2)*4+(c+2): 1}
                elif k == 4: # A00 + A01
                    # A00 is rows 0,1 cols 0,1. A01 is rows 0,1 cols 2,3.
                    # P[r][c] = A00[r][c] + A01[r][c].
                    # A00[r][c] is A[r][c].
                    # A01[r][c] is A[r][c+2].
                    return {r*4+c: 1, r*4+(c+2): 1}
                elif k == 5: # A10 - A00
                    # A10 is rows 2,3 cols 0,1. A00 is rows 0,1 cols 0,1.
                    # P[r][c] = A10[r][c] - A00[r][c].
                    return {(r+2)*4+c: 1, r*4+c: -1}
                elif k == 6: # A01 - A11
                    # A01 is rows 0,1 cols 2,3. A11 is rows 2,3 cols 2,3.
                    # P[r][c] = A01[r][c] - A11[r][c].
                    return {r*4+(c+2): 1, (r+2)*4+(c+2): -1}
            
            def expand_Q(k, r, c):
                # Similar for B.
                # B00: 0,0 to 1,1. B01: 0,2 to 1,3. B10: 2,0 to 3,1. B11: 2,2 to 3,3.
                if k == 0: # B00 + B11
                    return {r*4+c: 1, (r+2)*4+(c+2): 1}
                elif k == 1: # B00
                    return {r*4+c: 1}
                elif k == 2: # B01 - B11
                    # B01[r][c] is B[r][c+2]. B11[r][c] is B[r+2][c+2].
                    return {r*4+(c+2): 1, (r+2)*4+(c+2): -1}
                elif k == 3: # B10 - B00
                    # B10[r][c] is B[r+2][c]. B00[r][c] is B[r][c].
                    return {(r+2)*4+c: 1, r*4+c: -1}
                elif k == 4: # B11
                    return {(r+2)*4+(c+2): 1}
                elif k == 5: # B00 + B01
                    # B00[r][c] is B[r][c]. B01[r][c] is B[r][c+2].
                    return {r*4+c: 1, r*4+(c+2): 1}
                elif k == 6: # B10 + B11
                    # B10[r][c] is B[r+2][c]. B11[r][c] is B[r+2][c+2].
                    return {(r+2)*4+c: 1, (r+2)*4+(c+2): 1}

            # Now iterate
            for k in range(7):
                for j in range(7):
                    # Determine P and Q linear forms for step j
                    # P_step[j] is 2x2 matrix of coeffs for P elements.
                    # P_elem = sum_{pr, pc} P_step[j][pr][pc] * P[pr][pc]
                    
                    P_step = P_steps[j]
                    Q_step = Q_steps[j]
                    
                    # Build U vector
                    u_vec = {}
                    for pr in range(2):
                        for pc in range(2):
                            coeff_p = P_step[pr][pc]
                            if coeff_p != 0:
                                # Expand P[pr][pc]
                                p_expanded = expand_P(k, pr, pc)
                                for idx, coeff_a in p_expanded.items():
                                    val = coeff_p * coeff_a
                                    if val != 0:
                                        u_vec[idx] = u_vec.get(idx, 0) + val
                                        
                    v_vec = {}
                    for qr in range(2):
                        for qc in range(2):
                            coeff_q = Q_step[qr][qc]
                            if coeff_q != 0:
                                q_expanded = expand_Q(k, qr, qc)
                                for idx, coeff_b in q_expanded.items():
                                    val = coeff_q * coeff_b
                                    if val != 0:
                                        v_vec[idx] = v_vec.get(idx, 0) + val
                                        
                    # Fill U and V
                    for idx, val in u_vec.items():
                        U[idx][col] = val
                    for idx, val in v_vec.items():
                        V[idx][col] = val
                        
                    # Fill W
                    # Determine contribution to C
                    # m_j contributes to R (result of inner Strassen)
                    # R_coeff[j][rr][cc] is coeff of m_j in R[rr][cc]
                    for rr in range(2):
                        for cc in range(2):
                            coeff_r = R_coeff[j][rr][cc]
                            if coeff_r != 0:
                                # R[rr][cc] is part of block X_k
                                # X_k contributes to C blocks
                                # C_block[br][bc] = sum_k K[k][br][bc] * X_k
                                # Local position in X_k is rr, cc.
                                # Global position in C is (br*2+rr, bc*2+cc)
                                for br in range(2):
                                    for bc in range(2):
                                        coeff_k = K[k][br][bc]
                                        if coeff_k != 0:
                                            total_coeff = coeff_r * coeff_k
                                            if total_coeff != 0:
                                                gi = br * 2 + rr
                                                gl = bc * 2 + cc
                                                w_idx = gl * 4 + gi # W[k*n+i]
                                                W[w_idx][col] += total_coeff
                                                
                    col += 1
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
