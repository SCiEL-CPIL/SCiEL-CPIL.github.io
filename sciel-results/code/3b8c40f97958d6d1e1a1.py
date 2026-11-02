import time

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen coefficients for A (u=0..6)
    # u=0: a00 + a11
    # u=1: a10 + a11
    # u=2: a00
    # u=3: a11
    # u=4: a00 + a01
    # u=5: a10 - a00
    # u=6: a01 - a11
    U_coeffs = [
        {0: 1, 3: 1},       # u=0
        {2: 1, 3: 1},       # u=1
        {0: 1},             # u=2
        {3: 1},             # u=3
        {0: 1, 1: 1},       # u=4
        {2: 1, 0: -1},      # u=5
        {1: 1, 3: -1}       # u=6
    ]
    
    # Define Strassen coefficients for B block (v=0..6)
    # v=0: b0c0 + b1c1
    # v=1: b0c0
    # v=2: b1c0 - b1c1
    # v=3: b0c1 - b0c0
    # v=4: b1c1
    # v=5: b0c0 + b1c0
    # v=6: b1c1 + b0c1
    # Indices in V are j*p + k. j is row (0,1), k is col.
    # For block i, cols are 2i, 2i+1.
    # We will store relative coefficients and apply them in the loop.
    V_rel_coeffs = [
        {(0, 0): 1, (1, 1): 1},       # v=0
        {(0, 0): 1},                  # v=1
        {(1, 0): 1, (1, 1): -1},      # v=2
        {(0, 1): 1, (0, 0): -1},      # v=3
        {(1, 1): 1},                  # v=4
        {(0, 0): 1, (1, 0): 1},       # v=5
        {(1, 1): 1, (0, 1): 1}        # v=6
    ]
    
    # Define Reconstruction coefficients for C block (w=0..6)
    # C_00 = P0 + P3 - P4 - P6
    # C_01 = P2 + P4
    # C_10 = P1 + P6
    # C_11 = P0 - P1 + P2 + P5
    # Indices in W are k*n + i. k is col, i is row.
    # W_rel_coeffs[output_row, output_col] = {u: coeff}
    # output_row is i (0 or 1), output_col is relative 0 or 1.
    W_rel_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: -1},   # C_00
        {2: 1, 4: 1},                 # C_01
        {1: 1, 6: 1},                 # C_10
        {0: 1, 1: -1, 2: 1, 5: 1}     # C_11
    ]
    
    for i in range(4): # 4 blocks of 2 columns
        c0 = 2 * i
        c1 = 2 * i + 1
        for u in range(7):
            r = i * 7 + u
            # Fill U
            for idx, coeff in U_coeffs[u].items():
                U[idx][r] += coeff
            
            # Fill V
            for (j, k_rel), coeff in V_rel_coeffs[u].items():
                k_abs = c0 + k_rel
                v_idx = j * p + k_abs
                V[v_idx][r] += coeff
            
            # Fill W
            # W_rel_coeffs[row, col_rel]
            for row in range(2):
                for col_rel in range(2):
                    if u in W_rel_coeffs[row][col_rel]:
                        coeff = W_rel_coeffs[row][col_rel][u]
                        k_abs = c0 + col_rel
                        w_idx = k_abs * n + row
                        W[w_idx][r] += coeff
                        
    return {"U": U, "V": V, "W": W}
