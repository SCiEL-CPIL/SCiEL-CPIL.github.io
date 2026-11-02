import time
import numpy as np

# Dimensions: A is 2x2, B is 2x16, C is 2x16
# Tensor M_{2,2,16}
# Decomposed into 8 blocks of M_{2,2,2}
# Rank of M_{2,2,2} is 7 (Strassen)
# Total Rank = 8 * 7 = 56

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    n, m, p = 2, 2, 16
    blocks = p // 2
    total_rank = blocks * 7

    # Initialize matrices
    U = np.zeros((n * m, total_rank), dtype=np.int8)
    V = np.zeros((m * p, total_rank), dtype=np.int8)
    W = np.zeros((p * n, total_rank), dtype=np.int8)

    # Strassen coefficients for 2x2 matrix multiplication
    # Local indices within a 2x2 block:
    # U rows: 0:A00, 1:A01, 2:A10, 3:A11
    # V rows: 0:B00, 1:B01, 2:B10, 3:B11
    # W rows: 0:C00, 1:C01, 2:C10, 3:C11
    
    strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1 = (A00+A11)(B00+B11)
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}), # M2 = (A10+A11)B00
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}), # M3 = A00(B01-B11)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}), # M4 = A11(B10-B00)
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}), # M5 = (A00+A01)B11
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}), # M6 = (A10-A00)(B00+B01)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})  # M7 = (A01-A11)(B10+B11)
    ]

    for q in range(blocks):
        # Global indices for this block
        # B: row j (0,1), col 2q, 2q+1
        # V index: j*p + col
        b_local_to_global = {
            0: 0 * p + 2 * q,     # B[0, 2q]
            1: 0 * p + 2 * q + 1, # B[0, 2q+1]
            2: 1 * p + 2 * q,     # B[1, 2q]
            3: 1 * p + 2 * q + 1  # B[1, 2q+1]
        }

        # C: row i (0,1), col 2q, 2q+1
        # W index: col*n + row
        c_local_to_global = {
            0: (2 * q) * n + 0,     # C[0, 2q]
            1: (2 * q + 1) * n + 0, # C[0, 2q+1]
            2: (2 * q) * n + 1,     # C[1, 2q]
            3: (2 * q + 1) * n + 1  # C[1, 2q+1]
        }

        for s in range(7):
            r = q * 7 + s
            u_dict, v_dict, w_dict = strassen[s]

            for u_row, val in u_dict.items():
                U[u_row, r] = val

            for v_local, val in v_dict.items():
                V[b_local_to_global[v_local], r] = val

            for w_local, val in w_dict.items():
                W[c_local_to_global[w_local], r] = val

    # Verification with multiple test cases
    if time.time() < deadline - 10:
        test_cases = []
        np.random.seed(42)
        for trial in range(10):
            A = np.random.randint(-5, 5, size=(n, m))
            B = np.random.randint(-5, 5, size=(m, p))
            test_cases.append((A, B))
        
        # Also add edge cases
        test_cases.append((np.zeros((n, m)), np.ones((m, p))))
        test_cases.append((np.ones((n, m)), np.zeros((m, p))))
        test_cases.append((np.eye(n, m), np.ones((m, p))))
        test_cases.append((np.ones((n, m)), np.eye(m, p)))
        
        valid = True
        for A, B in test_cases:
            C = A @ B
            Af = A.flatten().astype(np.int64)
            Bf = B.flatten().astype(np.int64)
            
            u_prod = U.T @ Af
            v_prod = V.T @ Bf
            c_flat = W @ (u_prod * v_prod)
            
            c_expected = C.T.flatten()
            
            if not np.array_equal(c_flat, c_expected):
                valid = False
                break
        
        if not valid:
            # Fallback: Schoolbook
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=np.int8)
            V = np.zeros((m * p, rank), dtype=np.int8)
            W = np.zeros((p * n, rank), dtype=np.int8)
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
