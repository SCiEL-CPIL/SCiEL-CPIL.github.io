import numpy as np

def construct():
    n, m, p = 2, 2, 12
    num_blocks = p // m  # 6 blocks of width 2
    
    # Rank is 7 per block * 6 blocks = 42
    rank = num_blocks * 7
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen coefficients for 2x2 matrix multiplication
    # Each entry is a dict: {'U': [(idx, val), ...], 'V': [(idx, val), ...], 'W': [(idx, val), ...]}
    # Indices for U are (i, j) flattened to i*m + j (0: A00, 1: A01, 2: A10, 3: A11)
    # Indices for V are (j, k) flattened to j*p + k. 
    #   Within a block t, k_local is 0 or 1.
    #   V index for row j, col k_local in block t is j*p + (2*t + k_local)
    # Indices for W are (k, i) flattened to k*n + i.
    #   Within a block t, k_local is 0 or 1.
    #   W index for col k_local, row i in block t is (2*t + k_local)*n + i
    
    # We define the 7 multiplications M0..M6
    # U part depends only on A indices (0,1,2,3)
    # V part depends on B indices (j, k_local)
    # W part depends on C indices (k_local, i)
    
    # M0: (A00+A11)(B00+B11) -> C00+, C11+
    # M1: (A10+A11)B00       -> C10+, C11-
    # M2: A00(B01-B11)       -> C01+, C11+
    # M3: A11(B10-B00)       -> C00+, C10+
    # M4: (A00+A01)B11       -> C00-, C01+
    # M5: (A10-A00)(B00+B01) -> C11+
    # M6: (A01-A11)(B10+B11) -> C00+
    
    # Data structure for the 7 steps
    # step: (U_updates, V_updates, W_updates)
    # U_updates: list of (A_idx, coeff)
    # V_updates: list of (j, k_local, coeff)
    # W_updates: list of (k_local, i, coeff)
    
    strassen_steps = [
        # M0
        (
            [(0, 1), (3, 1)], 
            [(0, 0, 1), (1, 1, 1)], 
            [(0, 0, 1), (1, 1, 1)]
        ),
        # M1
        (
            [(2, 1), (3, 1)], 
            [(0, 0, 1)], 
            [(0, 1, 1), (1, 1, -1)]
        ),
        # M2
        (
            [(0, 1)], 
            [(0, 1, 1), (1, 1, -1)], 
            [(1, 0, 1), (1, 1, 1)]
        ),
        # M3
        (
            [(3, 1)], 
            [(1, 0, 1), (0, 0, -1)], 
            [(0, 0, 1), (0, 1, 1)]
        ),
        # M4
        (
            [(0, 1), (1, 1)], 
            [(1, 1, 1)], 
            [(0, 0, -1), (1, 0, 1)]
        ),
        # M5
        (
            [(2, 1), (0, -1)], 
            [(0, 0, 1), (0, 1, 1)], 
            [(1, 1, 1)]
        ),
        # M6
        (
            [(1, 1), (3, -1)], 
            [(1, 0, 1), (1, 1, 1)], 
            [(0, 0, 1)]
        ),
    ]
    
    for t in range(num_blocks):
        base_rank = t * 7
        for s in range(7):
            r_idx = base_rank + s
            u_defs, v_defs, w_defs = strassen_steps[s]
            
            # Fill U
            for a_idx, val in u_defs:
                U[a_idx, r_idx] += val
                
            # Fill V
            for j, k_loc, val in v_defs:
                v_idx = j * p + (2 * t + k_loc)
                V[v_idx, r_idx] += val
                
            # Fill W
            for k_loc, i, val in w_defs:
                w_idx = (2 * t + k_loc) * n + i
                W[w_idx, r_idx] += val
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
