import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 2, 16
    num_blocks = p // 2  # 8 blocks of 2 columns
    R = num_blocks * 7   # Total rank 56

    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)

    # Strassen coefficients for 2x2 x 2x2
    # Indices for A (2x2): 0:00, 1:01, 2:10, 3:11
    # Indices for B (2x2): 0:00, 1:01, 2:10, 3:11
    # Indices for C (2x2): 0:00, 1:01, 2:10, 3:11
    
    # 7 products P0..P6
    # U coefficients (for A)
    U_s = [
        {0: 1, 3: 1}, # P0 = (A00+A11)(B00+B11)
        {2: 1, 3: 1}, # P1 = (A10+A11) B00
        {0: 1},       # P2 = A00 (B01-B11)
        {3: 1},       # P3 = A11 (B10-B00)
        {0: 1, 1: 1}, # P4 = (A00+A01) B11
        {2: 1, 0: -1},# P5 = (A10-A00) (B00+B01)
        {1: 1, 3: -1} # P6 = (A01-A11) (B10+B11)
    ]

    # V coefficients (for B)
    V_s = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]

    # W coefficients (for C contribution)
    # C00 = P0 + P3 - P5 - P6
    # C01 = P4 + P6
    # C10 = P1 + P2
    # C11 = P0 - P1 + P2 + P4
    W_s = [
        {0: 1, 3: 1},   # P0 contributes to C00 and C11
        {2: 1, 3: -1},  # P1 contributes to C10 and C11 (-)
        {1: 1, 3: 1},   # P2 contributes to C01? No. 
                        # Let's re-verify Strassen W mapping.
                        # P0 -> C00, C11
                        # P1 -> C10, C11
                        # P2 -> C01, C11? No.
                        # Standard Strassen:
                        # C00 = P0 + P3 - P5 - P6
                        # C01 = P4 + P6
                        # C10 = P1 + P2
                        # C11 = P0 - P1 + P2 + P4
                        
                        # So:
                        # P0: {0: 1, 3: 1}
                        # P1: {2: 1, 3: -1}
                        # P2: {1: 1, 3: 1} -> Wait, P2 is in C10 and C11.
                        # C10 is index 2. C11 is index 3.
                        # My previous W_s[2] was {1:1, 3:1}. 
                        # Let's check P2 definition: A00(B01-B11).
                        # Does it contribute to C10? 
                        # C10 = A10 B00 + A11 B10.
                        # P1 = (A10+A11)B00.
                        # P2 = A00(B01-B11).
                        # P1 + P2 is NOT C10. 
                        
                        # Let's use the standard verified mapping:
                        # P0 = (A00+A11)(B00+B11)
                        # P1 = (A10+A11)(B00-B10)  <-- Variation
                        # ...
                        # The previous code's mapping was:
                        # U: {0:1, 3:1}, {2:1, 3:1}, {0:1}, {3:1}, {0:1, 1:1}, {2:1, 0:-1}, {1:1, 3:-1}
                        # V: {0:1, 3:1}, {0:1}, {1:1, 3:-1}, {2:1, 0:-1}, {3:1}, {0:1, 1:1}, {2:1, 3:1}
                        # W: {0:1, 3:1}, {2:1, 3:-1}, {1:1, 3:1}, {0:1, 2:1}, {0:-1, 1:1}, {3:1}, {0:1}
                        
                        # Let's stick to the verified coefficients from the previous run which passed tests.
        {1: 1, 3: 1},   
        {0: 1, 2: 1},   
        {0: -1, 1: 1},  
        {3: 1},         
        {0: 1}          
    ]

    # Re-define W_s to match the verified logic from previous run exactly to ensure correctness
    # Previous run W_contribs:
    # 0: {0: 1, 3: 1}
    # 1: {2: 1, 3: -1}
    # 2: {1: 1, 3: 1}
    # 3: {0: 1, 2: 1}
    # 4: {0: -1, 1: 1}
    # 5: {3: 1}
    # 6: {0: 1}
    
    W_s_final = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1}
    ]

    for q in range(num_blocks):
        # B indices for this block q (columns 2q, 2q+1)
        # B is 2x16. Flattened row-major: B[row, col]. Index = row * 16 + col.
        b_00 = 0 * p + 2 * q
        b_01 = 0 * p + (2 * q + 1)
        b_10 = 1 * p + 2 * q
        b_11 = 1 * p + (2 * q + 1)
        
        b_map = {0: b_00, 1: b_01, 2: b_10, 3: b_11}

        # C indices for this block q (columns 2q, 2q+1)
        # Output factor W[k*n+i, r]. W index = k * n + i.
        # C is 2x16. Flattened row-major? 
        # Contract says: W[k*n+i, r]. This corresponds to C[i, k].
        # So W row index is k*n+i.
        # For column k in C (which is 2q or 2q+1), and row i (0 or 1):
        c_00 = (2 * q) * n + 0   # i=0, k=2q
        c_01 = (2 * q + 1) * n + 0 # i=0, k=2q+1
        c_10 = (2 * q) * n + 1   # i=1, k=2q
        c_11 = (2 * q + 1) * n + 1 # i=1, k=2q+1
        
        c_map = {0: c_00, 1: c_01, 2: c_10, 3: c_11}

        for s in range(7):
            r = q * 7 + s
            for a_idx, coeff in U_s[s].items():
                U[a_idx, r] = coeff
            for local_b_idx, coeff in V_s[s].items():
                V[b_map[local_b_idx], r] = coeff
            for local_c_idx, coeff in W_s_final[s].items():
                W[c_map[local_c_idx], r] = coeff

    # Verification
    if time.time() < deadline - 1:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, size=(n, m))
        B_test = np.random.randint(-5, 5, size=(m, p))
        C_test = A_test @ B_test
        
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        UA = U.T @ A_flat
        VB = V.T @ B_flat
        terms = UA * VB
        C_decomp = W @ terms
        
        # W index is k*n+i. So C_decomp[ k*n + i ] = C[i, k].
        # We need to check if C_decomp matches C_test.T.flatten()
        # C_test.T is (p, n). Flatten is row-major: index = row * n + col = k * n + i.
        C_test_flat_T = C_test.T.flatten()
        
        if not np.array_equal(C_decomp, C_test_flat_T):
            # Fallback to Rank 64
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
