import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (x11+x22)(y11+y22)
    # M2 = (x21+x22)y11
    # M3 = x11(y12-y22)
    # M4 = x22(y21-y11)
    # M5 = (x11+x12)y22
    # M6 = (x21-x11)(y11+y12)
    # M7 = (x12-x22)(y21+y22)
    
    # Reconstruction:
    # z11 = M1 + M4 - M5 + M7
    # z12 = M3 + M5
    # z21 = M2 + M4
    # z22 = M1 - M2 + M3 + M6
    
    # Define the 7 (u_coeffs, v_coeffs, w_coeffs) for one Strassen block
    # u_coeffs: dict mapping (i,k) local to coeff
    # v_coeffs: dict mapping (k,j) local to coeff
    # w_coeffs: dict mapping (i,j) local to coeff (for output)
    
    strassen_terms = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
        # M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    ]
    
    # We have 4 blocks of multiplication:
    # Block 1: A_L * B_11 -> contributes to C[:, 0:2]
    # Block 2: A_R * B_21 -> contributes to C[:, 0:2]
    # Block 3: A_L * B_12 -> contributes to C[:, 2:4]
    # Block 4: A_R * B_22 -> contributes to C[:, 2:4]
    
    # Define offsets for A, B, C for each block
    # A is 2x4. A_L is cols 0-1, A_R is cols 2-3.
    # B is 4x4. B_11 is rows 0-1, cols 0-1. B_21 is rows 2-3, cols 0-1.
    #           B_12 is rows 0-1, cols 2-3. B_22 is rows 2-3, cols 2-3.
    # C is 2x4.
    
    blocks = [
        # (a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset)
        (0, 0, 0, 0, 0, 0), # A_L, B_11 -> C[:, 0:2]
        (0, 2, 2, 0, 0, 0), # A_R, B_21 -> C[:, 0:2]
        (0, 0, 0, 2, 0, 2), # A_L, B_12 -> C[:, 2:4]
        (0, 2, 2, 2, 0, 2), # A_R, B_22 -> C[:, 2:4]
    ]
    
    col = 0
    for (a_r, a_c, b_r, b_c, c_r, c_c) in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            # Populate U
            for (i, k), coeff in u_coeffs.items():
                u_idx = (a_r + i) * m + (a_c + k)
                U[u_idx][col] += coeff
            
            # Populate V
            for (k, j), coeff in v_coeffs.items():
                v_idx = (b_r + k) * p + (b_c + j)
                V[v_idx][col] += coeff
            
            # Populate W
            # W index is j*n + i
            # Here i is row of C, j is col of C
            for (i, j), coeff in w_coeffs.items():
                w_idx = (c_c + j) * n + (c_r + i)
                W[w_idx][col] += coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
