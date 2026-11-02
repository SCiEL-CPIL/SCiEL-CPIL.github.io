import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

def verify(U, V, W, n, m, p):
    rank = len(U[0])
    # Use numpy for faster verification
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # For each (i, j, k), we need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] == 1
    # We can compute this efficiently
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_reshaped[i, j, :], V_reshaped[j, k, :] * W_reshaped[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s
    
    # Schoolbook as default
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Construct Strassen-based decomposition for 4x4x4
    # Rank 28
    # Then combine for 4x4x9
    # Rank 28 + 28 + 16 = 72
    
    R = 72
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Helper to add a term
    # u_idx: i*m + j
    # v_idx: j*p + k
    # w_idx: k*n + i
    # coeff: value to add
    def add_term(i, j, k, c, col_idx):
        U[i*m + j][col_idx] += c
        V[j*p + k][col_idx] += c
        W[k*n + i][col_idx] += c

    # Strassen steps for 4x4
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # Step: (A_coeffs, B_coeffs)
    # A_coeffs: dict block -> coeff
    # B_coeffs: dict block -> coeff
    
    strassen_steps = [
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}), # M1
        ({'10': 1, '11': 1}, {'00': 1}),          # M2
        ({'00': 1}, {'01': 1, '11': -1}),         # M3
        ({'11': 1}, {'10': 1, '00': -1}),         # M4
        ({'00': 1, '01': 1}, {'11': 1}),          # M5
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}),# M6
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}) # M7
    ]
    
    # Output block positions for each Strassen step
    # C_00 = M1 - M3 + M5 - M7
    # C_01 = M3 + M4 + M7
    # C_10 = M2 + M4 - M6
    # C_11 = M1 - M2 + M5 + M6
    # Note: The signs and contributions need to be mapped to the 2x2 blocks.
    # Each M is a 2x2 matrix.
    # The contribution of M_s to C is a linear combination of M_s's blocks.
    
    # Let's define the contribution of each M_s to the 4 2x2 blocks of C.
    # contrib[s][block_idx] = coeff
    # block_idx: 0=00, 1=01, 2=10, 3=11
    
    contrib = [
        [1, 0, 0, 1],   # M1 to C00, C01, C10, C11? 
        # Wait, M1 = (A00+A11)(B00+B11).
        # C00 = M1 - M3 + M5 - M7.
        # C01 = M3 + M4 + M7.
        # C10 = M2 + M4 - M6.
        # C11 = M1 - M2 + M5 + M6.
        # So M1 contributes to C00 (1) and C11 (1).
        [1, 0, 0, 1],
        [0, 0, 1, -1],  # M2 to C10 (1), C11 (-1)
        [1, 1, 0, 0],   # M3 to C00 (-1)? No, C00 = M1 - M3. So -1.
        # Let's re-verify formulas.
        # C00 = M1 - M3 + M5 - M7
        # C01 = M3 + M4 + M7
        # C10 = M2 + M4 - M6
        # C11 = M1 - M2 + M5 + M6
        
        # M1: C00(1), C11(1)
        # M2: C10(1), C11(-1)
        # M3: C00(-1), C01(1)
        # M4: C01(1), C10(1)
        # M5: C00(1), C11(1)
        # M6: C10(-1), C11(1)
        # M7: C00(-1), C01(1)
    ]
    
    # Corrected contrib list:
    # Index 0: M1 -> C00(1), C11(1)
    # Index 1: M2 -> C10(1), C11(-1)
    # Index 2: M3 -> C00(-1), C01(1)
    # Index 3: M4 -> C01(1), C10(1)
    # Index 4: M5 -> C00(1), C11(1)
    # Index 5: M6 -> C10(-1), C11(1)
    # Index 6: M7 -> C00(-1), C01(1)
    
    contrib = [
        {0: 1, 3: 1},   # M1
        {2: 1, 3: -1},  # M2
        {0: -1, 1: 1},  # M3
        {1: 1, 2: 1},   # M4
        {0: 1, 3: 1},   # M5
        {2: -1, 3: 1},  # M6
        {0: -1, 1: 1}   # M7
    ]
    
    def add_strassen_block(b_offset, col_start):
        # b_offset: 0 for B cols 0-3, 4 for B cols 4-7
        col_idx = col_start
        for s in range(7):
            a_coeffs, b_coeffs = strassen_steps[s]
            # Expand 2x2 multiplication
            # M = A_s * B_s
            # A_s is 2x2, B_s is 2x2
            # M_00 = A_s00 B_s00 + A_s01 B_s10
            # M_01 = A_s00 B_s01 + A_s01 B_s11
            # M_10 = A_s10 B_s00 + A_s11 B_s10
            # M_11 = A_s10 B_s01 + A_s11 B_s11
            
            # We need to map A_s blocks to A indices.
            # A_s is formed by linear combo of A blocks.
            # A_block 00 -> A[0:2, 0:2]
            # A_block 01 -> A[0:2, 2:4]
            # A_block 10 -> A[2:4, 0:2]
            # A_block 11 -> A[2:4, 2:4]
            
            # For each of the 4 terms in the 2x2 product:
            for m_i in range(2):
                for m_j in range(2):
                    # Term is sum of products of A_inner and B_inner
                    # M[m_i, m_j] = sum_{k=0}^1 A_s[m_i, k] * B_s[k, m_j]
                    
                    # We need to decompose A_s[m_i, k] and B_s[k, m_j]
                    # A_s[m_i, k] is a linear combo of A entries.
                    # This is getting too complex for inline logic.
                    # Let's just generate the 4 scalar terms for the 2x2 product.
                    
                    # Actually, we can just add the 4 schoolbook terms for the 2x2 blocks.
                    # But A_s and B_s are themselves linear combos.
                    # So each "scalar" in M is a product of two linear combos.
                    # (Sum c_a A_a) * (Sum c_b B_b) = Sum c_a c_b A_a B_b.
                    # This expands to multiple terms.
                    
                    # Let's do it properly.
                    # For M[m_i, m_j]:
                    # It is sum_{k=0}^1 (Sum_{block_a} c_a A_{block_a}[m_i, k]) * (Sum_{block_b} c_b B_{block_b}[k, m_j])
                    
                    # We will iterate over k=0,1
                    for k in range(2):
                        # Get coefficients for A[m_i, k] and B[k, m_j]
                        a_terms = []
                        b_terms = []
                        
                        # A[m_i, k] comes from A_s[m_i, k]
                        # A_s is defined by a_coeffs.
                        # A_s[m_i, k] = sum_{block_a} a_coeffs[block_a] * A_{block_a}[m_i, k]
                        
                        for block_a, coeff_a in a_coeffs.items():
                            # A_{block_a}[m_i, k]
                            # block_a: 00, 01, 10, 11
                            # row_block = block_a[0]
                            # col_block = block_a[1]
                            # A global row = row_block * 2 + m_i
                            # A global col = col_block * 2 + k
                            row_a = int(block_a[0]) * 2 + m_i
                            col_a = int(block_a[1]) * 2 + k
                            a_terms.append((row_a, col_a, coeff_a))
                            
                        for block_b, coeff_b in b_coeffs.items():
                            # B_{block_b}[k, m_j]
                            # row_block = block_b[0]
                            # col_block = block_b[1]
                            # B global row = row_block * 2 + k
                            # B global col = b_offset + col_block * 2 + m_j
                            row_b = int(block_b[0]) * 2 + k
                            col_b = b_offset + int(block_b[1]) * 2 + m_j
                            b_terms.append((row_b, col_b, coeff_b))
                            
                        # Now expand product
                        for (ra, ca, coef_a) in a_terms:
                            for (rb, cb, coef_b) in b_terms:
                                # Term: coef_a * coef_b * A[ra, ca] * B[rb, cb]
                                # This contributes to C[m_i_block * 2 + m_i, m_j_block * 2 + m_j]?
                                # No, M is a 2x2 block of C.
                                # Which block of C?
                                # The Strassen step s contributes to specific C blocks with specific signs.
                                # M[m_i, m_j] is the (m_i, m_j) element of the 2x2 matrix M_s.
                                # This element corresponds to the same relative position in the target C blocks.
                                
                                # We need to know which C blocks M_s contributes to.
                                # contrib[s] is a dict {c_block_idx: sign}
                                # c_block_idx: 0=00, 1=01, 2=10, 3=11
                                # The position (m_i, m_j) in M_s maps to (m_i, m_j) in each target C block.
                                
                                for c_block_idx, sign in contrib[s].items():
                                    c_row_block = int(c_block_idx / 2)
                                    c_col_block = c_block_idx % 2
                                    
                                    c_row = c_row_block * 2 + m_i
                                    c_col = c_col_block * 2 + m_j
                                    
                                    # Add term
                                    # U index: c_row * m + ca ? No.
                                    # U is (n, m). n=4, m=4.
                                    # U[i, j]. i is row of A, j is col of A.
                                    # V is (m, p). m=4, p=9.
                                    # V[j, k]. j is col of A (row of B), k is col of B.
                                    # W is (p, n). p=9, n=4.
                                    # W[k, i]. k is col of B, i is row of A (col of C).
                                    
                                    # Our term is A[ra, ca] * B[rb, cb] * C[c_row, c_col]
                                    # So:
                                    # U index: ra * m + ca
                                    # V index: rb * p + cb
                                    # W index: cb * n + c_row
                                    
                                    coef = coef_a * coef_b * sign
                                    U[ra * m + ca][col_idx] += coef
                                    V[rb * p + cb][col_idx] += coef
                                    W[cb * n + c_row][col_idx] += coef
                        col_idx += 1
        return col_idx

    # Add 28 terms for B cols 0-3
    col_idx = add_strassen_block(0, 0)
    
    # Add 28 terms for B cols 4-7
    col_idx = add_strassen_block(4, col_idx)
    
    # Add 16 schoolbook terms for B col 8
    for i in range(4):
        for j in range(4):
            U[i * m + j][col_idx] = 1
            V[j * p + 8][col_idx] = 1
            W[8 * n + i][col_idx] = 1
            col_idx += 1
            
    # Verify
    if verify(U, V, W, n, m, p):
        best = {"U": U, "V": V, "W": W}
        best_rank = R
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        pass

    return best

