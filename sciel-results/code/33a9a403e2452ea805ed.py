import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    """
    Rank-49 decomposition for 2x2 * 2x14 matrix multiplication.
    Uses Strassen's algorithm for each of the 7 independent 2x2 blocks.
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 7 blocks
    rank = num_blocks * 7  # 49
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Correct Strassen factors for 2x2 block
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: 0:00, 1:01, 2:10, 3:11
    # C indices: 0:00, 1:01, 2:10, 3:11
    
    # U (coeffs for A)
    U_block = [
        [1, 0, 0, 1],   # M0: (a+d)
        [0, 0, 1, 1],   # M1: (c+d)
        [1, 0, 0, 0],   # M2: a
        [0, 0, 0, 1],   # M3: d
        [1, 1, 0, 0],   # M4: (a+b)
        [-1, 0, 1, 0],  # M5: (c-a)
        [0, 1, 0, -1]   # M6: (b-d)
    ]
    
    # V (coeffs for B)
    V_block = [
        [1, 0, 0, 1],   # M0: (e+h)
        [1, 0, 0, 0],   # M1: e
        [0, 1, 0, -1],  # M2: (f-h)
        [0, 0, 1, -1],  # M3: (g-e)
        [0, 0, 0, 1],   # M4: h
        [1, 1, 0, 0],   # M5: (e+f)
        [0, 0, 1, 1]    # M6: (g+h)
    ]
    
    # W (coeffs for C)
    # C00 = M0 - M3 + M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M5 + M6
    W_block = [
        [1, 0, 0, -1, 1, 0, -1],  # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 0, 0, 0, 1, 1]    # C11
    ]
    
    for r in range(num_blocks):
        # B indices for block r
        idx_b00 = 0 * p + (2 * r)
        idx_b01 = 0 * p + (2 * r + 1)
        idx_b10 = 1 * p + (2 * r)
        idx_b11 = 1 * p + (2 * r + 1)
        b_indices = [idx_b00, idx_b01, idx_b10, idx_b11]
        
        # W indices for C block r
        # C is 2x14. Flat index k*n + i.
        # Block r uses columns 2r, 2r+1.
        w_idx_c00 = (2 * r) * n + 0
        w_idx_c01 = (2 * r) * n + 1
        w_idx_c10 = (2 * r + 1) * n + 0
        w_idx_c11 = (2 * r + 1) * n + 1
        w_indices = [w_idx_c00, w_idx_c01, w_idx_c10, w_idx_c11]
        
        for s in range(7):
            c = 7 * r + s
            
            # Fill U
            for i in range(4):
                U[i][c] = float(U_block[s][i])
                
            # Fill V
            for i in range(4):
                V[b_indices[i]][c] = float(V_block[s][i])
                
            # Fill W
            for i in range(4):
                W[w_indices[i]][c] = float(W_block[i][s])
                
    return {"U": U, "V": V, "W": W}
