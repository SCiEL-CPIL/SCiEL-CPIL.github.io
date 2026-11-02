import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    
    # Rank 56 decomposition using 8 blocks of Strassen (rank 7)
    num_blocks = p // 2  # 8 blocks
    R = num_blocks * 7   # 56
    
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # Strassen coefficients for one 2x2 block
    U_coeffs = [
        {0: 1, 3: 1},      # M1
        {2: 1, 3: 1},      # M2
        {0: 1},            # M3
        {3: 1},            # M4
        {0: 1, 1: 1},      # M5
        {2: 1, 0: -1},     # M6
        {1: 1, 3: -1}      # M7
    ]
    
    V_coeffs = [
        {0: 1, 3: 1},      # M1
        {0: 1},            # M2
        {1: 1, 3: -1},     # M3
        {2: 1, 0: -1},     # M4
        {3: 1},            # M5
        {0: 1, 1: 1},      # M6
        {2: 1, 3: 1}       # M7
    ]
    
    W_contribs = [
        {0: 1, 3: 1},      # M1 -> C00, C11
        {2: 1, 3: -1},     # M2 -> C10, C11
        {1: 1, 3: 1},      # M3 -> C01, C11
        {0: 1, 2: 1},      # M4 -> C00, C10
        {0: -1, 1: 1},     # M5 -> C00, C01
        {3: 1},            # M6 -> C11
        {0: 1}             # M7 -> C00
    ]
    
    for q in range(num_blocks):
        b_00 = 0 * p + 2 * q
        b_01 = 0 * p + (2 * q + 1)
        b_10 = 1 * p + 2 * q
        b_11 = 1 * p + (2 * q + 1)
        
        c_00 = (2 * q) * n + 0
        c_01 = (2 * q + 1) * n + 0
        c_10 = (2 * q) * n + 1
        c_11 = (2 * q + 1) * n + 1
        
        b_map = {0: b_00, 1: b_01, 2: b_10, 3: b_11}
        c_map = {0: c_00, 1: c_01, 2: c_10, 3: c_11}
        
        for s in range(7):
            r = q * 7 + s
            
            for a_idx, coeff in U_coeffs[s].items():
                U[a_idx, r] = coeff
                
            for local_idx, coeff in V_coeffs[s].items():
                V[b_map[local_idx], r] = coeff
                
            for local_out_idx, coeff in W_contribs[s].items():
                W[c_map[local_out_idx], r] = coeff

    # Verify with test cases for robustness (with deadline guard)
    valid = True
    for seed in range(3):
        if time.time() > deadline:
            break
        np.random.seed(seed)
        A_test = np.random.randint(-5, 5, size=(n, m))
        B_test = np.random.randint(-5, 5, size=(m, p))
        C_test = A_test @ B_test
        
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        UA = U.T @ A_flat
        VB = V.T @ B_flat
        terms = UA * VB
        C_decomp = W @ terms
        C_test_flat = C_test.T.flatten()
        
        if not np.allclose(C_decomp, C_test_flat, rtol=1e-9, atol=1e-9):
            valid = False
            break
    
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
