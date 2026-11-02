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
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C local indices (within 2x2 block): 0=C00, 1=C10, 2=C01, 3=C11
    # (matching W[k*n+i] convention: C[i,k] -> index k*2+i)
    
    # 7 Strassen products
    U_s = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    V_s = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    W_s = [
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
            for local_c_idx, coeff in W_s[s].items():
                W[c_map[local_c_idx], r] = coeff

    # Verification with multiple random tests
    if time.time() < deadline - 2:
        valid = True
        for seed in range(3):
            np.random.seed(seed)
            A_test = np.random.randint(-3, 4, size=(n, m))
            B_test = np.random.randint(-3, 4, size=(m, p))
            C_test = A_test @ B_test
            
            A_flat = A_test.flatten()
            B_flat = B_test.flatten()
            
            UA = U.T @ A_flat
            VB = V.T @ B_flat
            terms = UA * VB
            C_decomp = W @ terms
            
            # W index k*n+i corresponds to C[i,k], so compare with C_test.T.flatten()
            C_test_flat_T = C_test.T.flatten()
            
            if not np.array_equal(C_decomp, C_test_flat_T):
                valid = False
                break
        
        if not valid:
            # Fallback to trivial rank 64
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
