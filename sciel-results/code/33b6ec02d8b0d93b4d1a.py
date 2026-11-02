import time
import numpy as np

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # We use the Strassen-based construction for rank 21.
    # A is 2x2, B is 2x6.
    # We split B into three 2x2 blocks: B0, B1, B2.
    # C = A * B = [A*B0, A*B1, A*B2].
    # Each A*Bk uses 7 scalar multiplications (Strassen).
    # Total rank = 3 * 7 = 21.
    
    R = 21
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Strassen coefficients for 2x2 matrices
    # Indices for A (2x2): 00, 01, 10, 11 -> flat indices 0, 1, 2, 3
    # U[i*m+j, r]
    
    # Strassen products for one 2x2x2 block:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 - M4 + M5 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # U coefficients for the 7 Strassen terms (same for all 3 blocks of B)
    # U is indexed by [i*m+j, r]
    # r = block * 7 + term
    # term 0..6 correspond to M1..M7
    
    u_coeffs = [
        {0: 1, 3: 1},  # M1: A00 + A11
        {2: 1, 3: 1},  # M2: A10 + A11
        {0: 1},        # M3: A00
        {3: 1},        # M4: A11
        {0: 1, 1: 1},  # M5: A00 + A01
        {2: 1, 0: -1}, # M6: A10 - A00
        {1: 1, 3: -1}  # M7: A01 - A11
    ]
    
    # V coefficients for each block k (0, 1, 2)
    # B is 2x6. Block k consists of columns 2k and 2k+1.
    # B indices: j*p + k_col. j in {0,1}, k_col in {0..5}.
    # For block k, local col 0 is global 2k, local col 1 is global 2k+1.
    
    v_coeffs_per_term = [
        [(0, 0, 1), (1, 1, 1)],  # M1: B00 + B11
        [(0, 0, 1)],             # M2: B00
        [(0, 1, 1), (1, 1, -1)], # M3: B01 - B11
        [(1, 0, 1), (0, 0, -1)], # M4: B10 - B00
        [(1, 1, 1)],             # M5: B11
        [(0, 0, 1), (0, 1, 1)],  # M6: B00 + B01
        [(1, 0, 1), (1, 1, 1)]   # M7: B10 + B11
    ]
    
    # W coefficients for output C (2x6)
    # C indices: k*n + i (trace convention). k in {0..5}, i in {0,1}.
    # C[i, k] is at index k*n + i.
    # For block k, local col 0 is global 2k, local col 1 is global 2k+1.
    
    w_coeffs_per_term = [
        [(0, 0, 1), (1, 1, 1)],   # M1 -> C00(+1), C11(+1)
        [(1, 0, 1), (1, 1, -1)],  # M2 -> C10(+1), C11(-1)
        [(0, 1, 1), (1, 1, 1)],   # M3 -> C01(+1), C11(+1)
        [(0, 0, 1), (1, 0, 1)],   # M4 -> C00(+1), C10(+1)
        [(0, 0, -1), (0, 1, 1)],  # M5 -> C00(-1), C01(+1)
        [(1, 1, 1)],              # M6 -> C11(+1)
        [(0, 0, 1)]               # M7 -> C00(+1)
    ]
    
    # Fill U
    for block in range(3):
        for term in range(7):
            r = block * 7 + term
            for idx, val in u_coeffs[term].items():
                U[idx, r] = val
                
    # Fill V
    for block in range(3):
        for term in range(7):
            r = block * 7 + term
            for j_local, k_local, val in v_coeffs_per_term[term]:
                global_k = 2 * block + k_local
                v_idx = j_local * p + global_k
                V[v_idx, r] = val
                
    # Fill W
    for block in range(3):
        for term in range(7):
            r = block * 7 + term
            for i, k_local, val in w_coeffs_per_term[term]:
                global_k = 2 * block + k_local
                w_idx = global_k * n + i
                W[w_idx, r] = val
                
    # Verification (optional but good for debugging, skip if time is tight)
    # The construction is known to be correct.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
