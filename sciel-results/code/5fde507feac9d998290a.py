import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x8 matrix multiplication.
    The minimal rank for M_{2,2,8} is 28, achieved by applying Strassen's algorithm
    independently to each of the four 2x2 blocks of the second matrix.
    """
    n, m, p = DIMENSIONS  # 2, 2, 8
    rank = 28
    
    # Strassen parameters for 2x2x2 multiplication
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # U vectors (coefficients for A)
    # Indices: 0:a, 1:b, 2:c, 3:d
    U_strassen = [
        [1, 0, 0, 1],  # M1: a+d
        [0, 0, 1, 1],  # M2: c+d
        [1, 0, 0, 0],  # M3: a
        [0, 0, 0, 1],  # M4: d
        [1, 1, 0, 0],  # M5: a+b
        [-1, 0, 1, 0], # M6: c-a
        [0, 1, 0, -1]  # M7: b-d
    ]
    
    # V vectors (coefficients for B)
    # Indices: 0:e, 1:f, 2:g, 3:h
    V_strassen = [
        [1, 0, 0, 1],  # M1: e+h
        [1, 0, 0, 0],  # M2: e
        [0, 1, 0, -1], # M3: f-h
        [-1, 0, 1, 0], # M4: g-e
        [0, 0, 0, 1],  # M5: h
        [1, 1, 0, 0],  # M6: e+f
        [0, 0, 1, 1]   # M7: g+h
    ]
    
    # W vectors (coefficients for C)
    # Indices: 0:c00, 1:c01, 2:c10, 3:c11
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    W_strassen = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0]    # M7
    ]
    
    num_blocks = p // 2  # 4 blocks of 2x2
    rank = num_blocks * 7 # 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            # Fill U
            for i in range(n * m):
                U[i][col] = U_strassen[r][i]
                
            # Fill V
            # B is m x p (2 x 8)
            # Global index = row * p + col_idx
            # Local 0: row 0, col 2t -> 0*8 + 2t = 2t
            # Local 1: row 0, col 2t+1 -> 0*8 + 2t+1 = 2t+1
            # Local 2: row 1, col 2t -> 1*8 + 2t = 8+2t
            # Local 3: row 1, col 2t+1 -> 1*8 + 2t+1 = 8+2t+1
            v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
            for l in range(4):
                V[v_indices[l]][col] = V_strassen[r][l]
                
            # Fill W
            # C is n x p (2 x 8)
            # Convention W[k*n + i, r] where k is col, i is row
            # Local 0: c[0][2t] -> i=0, k=2t -> 2t*2 + 0 = 4t
            # Local 1: c[0][2t+1] -> i=0, k=2t+1 -> (2t+1)*2 + 0 = 4t+2
            # Local 2: c[1][2t] -> i=1, k=2t -> 2t*2 + 1 = 4t+1
            # Local 3: c[1][2t+1] -> i=1, k=2t+1 -> (2t+1)*2 + 1 = 4t+3
            w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for l in range(4):
                W[w_indices[l]][col] = W_strassen[r][l]
                
            col += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
