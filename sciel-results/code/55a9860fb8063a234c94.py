import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # n=2, m=2, p=16

    num_blocks = p // 2  # 8 blocks of 2 columns each
    R = num_blocks * 7   # 56

    U = np.zeros((n * m, R), dtype=int)  # (4, 56)
    V = np.zeros((m * p, R), dtype=int)  # (32, 56)
    W = np.zeros((p * n, R), dtype=int)  # (32, 56)

    # Standard Strassen for 2x2 matmul
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B local: 0=B00, 1=B01, 2=B10, 3=B11
    # C local: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6

    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    for q in range(num_blocks):
        # B block q: columns 2q, 2q+1
        # B[0, 2q] -> V index 0*16 + 2q = 2q
        # B[0, 2q+1] -> V index 0*16 + 2q+1 = 2q+1
        # B[1, 2q] -> V index 1*16 + 2q = 16+2q
        # B[1, 2q+1] -> V index 1*16 + 2q+1 = 16+2q+1
        
        b_map = {0: 2*q, 1: 2*q+1, 2: 16+2*q, 3: 16+2*q+1}
        
        # C block q: columns 2q, 2q+1
        # C[0, 2q] -> W index 2q*2 + 0 = 4q
        # C[0, 2q+1] -> W index (2q+1)*2 + 0 = 4q+2
        # C[1, 2q] -> W index 2q*2 + 1 = 4q+1
        # C[1, 2q+1] -> W index (2q+1)*2 + 1 = 4q+3
        
        c_map = {0: 4*q, 1: 4*q+2, 2: 4*q+1, 3: 4*q+3}

        for s in range(7):
            r = q * 7 + s
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]
            
            for a_idx, coeff in u_coeffs.items():
                U[a_idx, r] = coeff
            for local_b, coeff in v_coeffs.items():
                V[b_map[local_b], r] = coeff
            for local_c, coeff in w_coeffs.items():
                W[c_map[local_c], r] = coeff

    # Verify correctness with exact integer arithmetic
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, size=(n, m))
        B_test = np.random.randint(-5, 5, size=(m, p))
        C_test = A_test @ B_test
        
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        UA = U.T @ A_flat  # (R,)
        VB = V.T @ B_flat  # (R,)
        terms = UA * VB
        C_decomp = W @ terms
        
        C_test_flat = C_test.T.flatten()  # W[k*n+i, r] convention: c = k*n+i
        
        if not np.array_equal(C_decomp, C_test_flat):
            # Fallback to schoolbook rank 64
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)
            
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        U[u_idx, column] = 1
                        V[v_idx, column] = 1
                        W[w_idx, column] = 1
                        column += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
