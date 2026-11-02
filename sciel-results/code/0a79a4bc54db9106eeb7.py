import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 8
    R = num_blocks * 7   # 56

    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)

    # Strassen decomposition for 2x2 x 2x2 -> 2x2
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B local: 0=B00, 1=B01, 2=B10, 3=B11
    # C local: 0=C00, 1=C01, 2=C10, 3=C11

    U_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]

    V_coeffs = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]

    W_contribs = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1}
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

    # Verify with a small random integer test
    if time.time() < deadline - 1:
        np.random.seed(12345)
        A_test = np.random.randint(-3, 3, size=(n, m))
        B_test = np.random.randint(-3, 3, size=(m, p))
        C_test = A_test @ B_test

        A_flat = A_test.flatten()
        B_flat = B_test.flatten()

        UA = U.T @ A_flat
        VB = V.T @ B_flat
        terms = UA * VB
        C_decomp = W @ terms

        # W[k*n+i, r] corresponds to C[i, k], so C_decomp is C.T flattened
        C_test_flat = C_test.T.flatten()

        if not np.array_equal(C_decomp, C_test_flat):
            # Fallback: schoolbook rank = n*m*p = 64
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
