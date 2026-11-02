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
    
    # Correct Strassen coefficients for C = A * B where A, B are 2x2
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    #
    # Using the variant matching the U, V below:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M6
    
    U_coeffs = [
        {0: 1, 3: 1},      # M1: a00 + a11
        {2: 1, 3: 1},      # M2: a10 + a11
        {0: 1},            # M3: a00
        {3: 1},            # M4: a11
        {0: 1, 1: 1},      # M5: a00 + a01
        {2: 1, 0: -1},     # M6: a10 - a00
        {1: 1, 3: -1}      # M7: a01 - a11
    ]
    
    V_coeffs = [
        {0: 1, 3: 1},      # M1: b00 + b11
        {0: 1},            # M2: b00
        {1: 1, 3: -1},     # M3: b01 - b11
        {2: 1, 0: -1},     # M4: b10 - b00
        {3: 1},            # M5: b11
        {0: 1, 1: 1},      # M6: b00 + b01
        {2: 1, 3: 1}       # M7: b10 + b11
    ]
    
    # W: which C entries each M contributes to, with coefficients
    # c_map: 0->c00, 1->c01, 2->c10, 3->c11
    W_contribs = [
        {0: 1, 3: 1},      # M1: +c00, +c11
        {2: 1},            # M2: +c10
        {1: 1, 3: -1},     # M3: +c01, -c11
        {0: -1, 2: 1},     # M4: -c00, +c10
        {1: 1, 3: 1},      # M5: +c01, +c11
        {0: 1, 3: -1},     # M6: +c00, -c11
        {0: 1}             # M7: +c00
    ]
    
    for q in range(num_blocks):
        if time.time() > deadline:
            break
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

    # Verify with multiple test cases for robustness
    valid = True
    for seed in range(10):
        if time.time() > deadline - 5:
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
