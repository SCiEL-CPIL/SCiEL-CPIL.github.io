import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We are decomposing M_{4, 4, 6}
    # Using 12 Strassen blocks of size 2x2x2
    
    # Global dimensions
    U_rows = n * m  # 16
    V_rows = m * p  # 24
    W_rows = p * n  # 24
    
    rank = 84
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # X is 2x2, Y is 2x2, Z is 2x2
    # Z = X Y
    # 7 terms
    
    # Term 1: (X00 + X11) * (Y00 + Y11)
    # Term 2: (X10 + X11) * Y00
    # Term 3: X00 * (Y10 - Y11)
    # Term 4: X11 * (Y01 - Y00)
    # Term 5: (X00 + X10) * Y11
    # Term 6: (X10 - X00) * (Y00 + Y10)
    # Term 7: (X11 - X10) * (Y01 + Y11)
    
    # Reconstruction:
    # Z00 = T1 + T4 - T5 + T7
    # Z01 = T3 + T5
    # Z10 = T2 + T4
    # Z11 = T1 - T2 + T3 + T6
    
    # Define the linear forms for X (A block) and Y (B block)
    # X indices: 00, 01, 10, 11 (row, col within 2x2 block)
    # Y indices: 00, 01, 10, 11
    
    # For each sub-problem (i, j, k) where i in 0..1, j in 0..2, k in 0..1
    # A block is A[i, k]
    # B block is B[k, j]
    # C block is C[i, j]
    
    for i in range(2):
        for j in range(3):
            for k in range(2):
                # Define local indices for A block (i, k)
                # A global row: i*2 + r, col: k*2 + c
                # A block rows: [i*2, i*2+1]
                # A block cols: [k*2, k*2+1]
                
                # Define local indices for B block (k, j)
                # B global row: k*2 + r, col: j*2 + c
                # B block rows: [k*2, k*2+1]
                # B block cols: [j*2, j*2+1]
                
                # Define local indices for C block (i, j)
                # C global row: i*2 + r, col: j*2 + c
                # C block rows: [i*2, i*2+1]
                # C block cols: [j*2, j*2+1]
                
                # We add 7 terms for this sub-problem
                
                # Term 1
                # X: X00 + X11 -> A[i*2, k*2] + A[i*2+1, k*2+1]
                # Y: Y00 + Y11 -> B[k*2, j*2] + B[k*2+1, j*2+1]
                # Z: Z00 += 1, Z11 += 1
                
                # Helper to add a term
                # u_coeffs: dict (local_r, local_c) -> coeff
                # v_coeffs: dict (local_r, local_c) -> coeff
                # w_coeffs: dict (local_r, local_c) -> coeff
                
                def add_term(u_coeff, v_coeff, w_coeff, i, j, k):
                    nonlocal col
                    # U
                    for (lr, lc), val in u_coeff.items():
                        gr = i * 2 + lr
                        gc = k * 2 + lc
                        U[gr * m + gc][col] += val
                    # V
                    for (lr, lc), val in v_coeff.items():
                        gr = k * 2 + lr
                        gc = j * 2 + lc
                        V[gr * p + gc][col] += val
                    # W
                    for (lr, lc), val in w_coeff.items():
                        gr = j * 2 + lr  # Note: W index is k*n+i? No, W[k*n+i, r]
                        # Task says: W[k*n+i, r]
                        # C is p x n? No, C is n x p?
                        # "multiplying a 4 by 4 matrix by a 4 by 6 matrix"
                        # A is 4x4, B is 4x6. C is 4x6.
                        # Output factor W[k*n+i, r].
                        # k is index of B col? i is index of A row?
                        # C[i, j] = sum A[i, k] B[k, j].
                        # So W index corresponds to C[i, j].
                        # The contract says W[k*n+i, r].
                        # This implies W is indexed by (k, i) where k is col of C?
                        # If C is 4x6, then i is row (0..3), k is col (0..5)?
                        # "W[k*n+i, r]" with n=4.
                        # If k is col index (0..5) and i is row index (0..3), then k*4+i covers 0..23.
                        # So W row index = col_C * n + row_C.
                        # My C block (i, j) has row i (0..1 in block, 0..3 global) and col j (0..2 in block, 0..5 global).
                        # Global row: i*2 + lr.
                        # Global col: j*2 + lc.
                        # W row index: (j*2 + lc) * n + (i*2 + lr).
                        
                        gw_row = i * 2 + lr
                        gw_col = j * 2 + lc
                        w_idx = gw_col * n + gw_row
                        W[w_idx][col] += val
                    col += 1

                # Term 1
                u1 = {(0,0): 1, (1,1): 1}
                v1 = {(0,0): 1, (1,1): 1}
                w1 = {(0,0): 1, (1,1): 1}
                add_term(u1, v1, w1, i, j, k)
                
                # Term 2
                u2 = {(1,0): 1, (1,1): 1}
                v2 = {(0,0): 1}
                w2 = {(1,0): 1}
                add_term(u2, v2, w2, i, j, k)
                
                # Term 3
                u3 = {(0,0): 1}
                v3 = {(1,0): 1, (1,1): -1}
                w3 = {(0,1): 1}
                add_term(u3, v3, w3, i, j, k)
                
                # Term 4
                u4 = {(1,1): 1}
                v4 = {(0,1): 1, (0,0): -1}
                w4 = {(0,0): 1, (1,0): 1}
                add_term(u4, v4, w4, i, j, k)
                
                # Term 5
                u5 = {(0,0): 1, (1,0): 1}
                v5 = {(1,1): 1}
                w5 = {(0,0): -1, (0,1): 1}
                add_term(u5, v5, w5, i, j, k)
                
                # Term 6
                u6 = {(1,0): 1, (0,0): -1}
                v6 = {(0,0): 1, (1,0): 1}
                w6 = {(1,1): 1}
                add_term(u6, v6, w6, i, j, k)
                
                # Term 7
                u7 = {(1,1): 1, (1,0): -1}
                v7 = {(0,1): 1, (1,1): 1}
                w7 = {(0,0): 1}
                add_term(u7, v7, w7, i, j, k)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
