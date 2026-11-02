import numpy as np
import time

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen factors for A (2x2)
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    A_factors = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1]  # M7: A01 - A11
    ]
    
    # Strassen factors for B (2x2 block)
    # B indices in block: 00, 01, 10, 11
    # Corresponds to B[0, jL], B[0, jR], B[1, jL], B[1, jR]
    B_factors = [
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [-1, 0, 1, 0], # M4: B10 - B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1]   # M7: B10 + B11
    ]
    
    # W coefficients for C
    # C indices: (row, col)
    # Convention: W[k*n + i, r] where k is col (0..15), i is row (0..1)
    # So index = col * 2 + row
    
    # First, build the raw W without Hadamard mixing
    W_raw = np.zeros((p * n, rank))
    
    for b in range(8):
        if time.time() > deadline:
            break
            
        jL = 2 * b
        jR = 2 * b + 1
        
        for s in range(7):
            col = b * 7 + s
            
            # Fill U
            for idx, val in enumerate(A_factors[s]):
                U[idx, col] = val
                
            # Fill V
            # B indices mapping:
            # 0: B[0, jL] -> row 0, col jL -> index 0*16 + jL
            # 1: B[0, jR] -> row 0, col jR -> index 0*16 + jR
            # 2: B[1, jL] -> row 1, col jL -> index 1*16 + jL
            # 3: B[1, jR] -> row 1, col jR -> index 1*16 + jR
            V_indices = [0 * 16 + jL, 0 * 16 + jR, 1 * 16 + jL, 1 * 16 + jR]
            for idx, val in enumerate(B_factors[s]):
                V[V_indices[idx], col] = val
                
            # Fill W_raw
            # W indices: W[col * 2 + row, col_rank]
            
            w_C0L = jL * 2 + 0
            w_C0R = jR * 2 + 0
            w_C1L = jL * 2 + 1
            w_C1R = jR * 2 + 1
            
            if s == 0: # M1
                W_raw[w_C0L, col] += 1
                W_raw[w_C1R, col] += 1
            elif s == 1: # M2
                W_raw[w_C1L, col] += 1
                W_raw[w_C1R, col] -= 1
            elif s == 2: # M3
                W_raw[w_C0R, col] += 1
                W_raw[w_C1R, col] += 1
            elif s == 3: # M4
                W_raw[w_C0L, col] += 1
                W_raw[w_C1L, col] += 1
            elif s == 4: # M5
                W_raw[w_C0L, col] -= 1
                W_raw[w_C0R, col] += 1
            elif s == 5: # M6
                W_raw[w_C1R, col] += 1
            elif s == 6: # M7
                W_raw[w_C0L, col] += 1
    
    # The Hadamard mixing approach was found to be invalid due to scaling constraints
    # (introducing non-integer entries). We stick to the direct block decomposition
    # which is valid and achieves rank 56.
    W = W_raw
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
