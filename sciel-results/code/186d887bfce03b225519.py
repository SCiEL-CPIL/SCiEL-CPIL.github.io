"""Matrix multiplication 2x2 by 2x7 - rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 7]

def construct():
    """
    Construct a tensor decomposition for M(2,2,7).
    
    Uses Strassen for three 2x2 blocks and schoolbook for the last 2x1 column.
    Rank = 3 * 7 + 4 = 25.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=7
    deadline = time.time() + 140  # safety margin before 150s kill

    rank = 25
    U = [[0.0] * rank for _ in range(n * m)]   # (4, 25)
    V = [[0.0] * rank for _ in range(m * p)]   # (14, 25)
    W = [[0.0] * rank for _ in range(p * n)]   # (14, 25)

    r = 0
    
    # Strassen parameters for 2x2 * 2x2
    # M1..M7
    # U coeffs (a00, a01, a10, a11)
    U_strassen = [
        [1, 0, 0, 1],  # M1: a00 + a11
        [0, 0, 1, 1],  # M2: a10 + a11
        [1, 0, 0, 0],  # M3: a00
        [0, 0, 0, 1],  # M4: a11
        [1, 1, 0, 0],  # M5: a00 + a01
        [-1, 0, 1, 0], # M6: a10 - a00
        [0, 1, 0, -1]  # M7: a01 - a11
    ]
    
    # V coeffs for local B (b00, b01, b10, b11) -> global indices
    # b00: 0, b01: 1, b10: 2, b11: 3 (local)
    V_strassen_local = [
        [1, 0, 0, 1],  # M1: b00 + b11
        [1, 0, 0, 0],  # M2: b00
        [0, 1, 0, -1], # M3: b01 - b11
        [0, 0, 1, -1], # M4: b10 - b00
        [0, 0, 0, 1],  # M5: b11
        [1, 1, 0, 0],  # M6: b00 + b01
        [0, 0, 1, 1]   # M7: b10 + b11
    ]
    
    # W coeffs for local C (c00, c01, c10, c11) -> global indices
    # c00: 0, c01: 1, c10: 2, c11: 3 (local)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    W_strassen_local = [
        [1, 0, 0, 1],  # M1: C00, C11
        [0, 0, 1, -1], # M2: C10, -C11
        [0, 1, 0, 1],  # M3: C01, C11
        [1, 0, 1, 0],  # M4: C00, C10
        [-1, 1, 0, 0], # M5: -C00, C01
        [0, 0, 0, 1],  # M6: C11
        [1, 0, 0, 0]   # M7: C00
    ]
    
    # Process 3 blocks of 2 columns each
    for block in range(3):
        col_start = block * 2
        for k in range(7):
            # U
            for i in range(4):
                U[i][r] = U_strassen[k][i]
            
            # V
            # Map local (0,1,2,3) to global (0, 1, 2, 3) + offset
            # local 0: b00 -> global 0*7 + col_start = col_start
            # local 1: b01 -> global 0*7 + col_start + 1
            # local 2: b10 -> global 1*7 + col_start
            # local 3: b11 -> global 1*7 + col_start + 1
            V_indices = [
                0 * p + col_start,
                0 * p + col_start + 1,
                1 * p + col_start,
                1 * p + col_start + 1
            ]
            for l in range(4):
                V[V_indices[l]][r] = V_strassen_local[k][l]
                
            # W
            # Map local (0,1,2,3) to global
            # local 0: c00 -> global (col_start)*2 + 0
            # local 1: c01 -> global (col_start+1)*2 + 0
            # local 2: c10 -> global (col_start)*2 + 1
            # local 3: c11 -> global (col_start+1)*2 + 1
            W_indices = [
                col_start * n + 0,
                (col_start + 1) * n + 0,
                col_start * n + 1,
                (col_start + 1) * n + 1
            ]
            for l in range(4):
                W[W_indices[l]][r] = W_strassen_local[k][l]
                
            r += 1
            
    # Process last column (2x1) with schoolbook
    # C06 = a00 b06 + a01 b16
    # C16 = a10 b06 + a11 b16
    
    # Term 1: a00 b06 -> C06
    U[0][r] = 1.0 # a00
    V[0 * p + 6][r] = 1.0 # b06
    W[6 * n + 0][r] = 1.0 # C06
    r += 1
    
    # Term 2: a01 b16 -> C06
    U[1][r] = 1.0 # a01
    V[1 * p + 6][r] = 1.0 # b16
    W[6 * n + 0][r] = 1.0 # C06
    r += 1
    
    # Term 3: a10 b06 -> C16
    U[2][r] = 1.0 # a10
    V[0 * p + 6][r] = 1.0 # b06
    W[6 * n + 1][r] = 1.0 # C16
    r += 1
    
    # Term 4: a11 b16 -> C16
    U[3][r] = 1.0 # a11
    V[1 * p + 6][r] = 1.0 # b16
    W[6 * n + 1][r] = 1.0 # C16
    r += 1

    return {"U": U, "V": V, "W": W}
