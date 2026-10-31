<<<<<<< SEARCH
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 12.0  # Safety margin well under 120s

    n, m, p = DIMENSIONS
    rank = 49  # 7 * 7: outer Strassen (7 terms) x inner Strassen (7 terms)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2 coefficients (7 terms each for A, B, C linear forms)
    # a_coeffs[s][i][j]: coefficient of A[i][j] in s-th outer/inner A-form
    # b_coeffs[s][j][k]: coefficient of B[j][k] in s-th outer/inner B-form
    # c_coeffs[s][i][k]: coefficient of C[i][k] in s-th outer/inner C-form
    #
    # Standard Strassen:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # C00 = M0 - M3 + M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M5

    a_coeffs = [
        [[1, 0], [0, 1]],   # s=0: A00 + A11
        [[0, 0], [1, 1]],   # s=1: A10 + A11
        [[1, 0], [0, 0]],   # s=2: A00
        [[0, 0], [0, 1]],   # s=3: A11
        [[1, 1], [0, 0]],   # s=4: A00 + A01
        [[-1, 0], [1, 0]],  # s=5: A10 - A00
        [[0, 1], [0, -1]],  # s=6: A01 - A11
    ]

    b_coeffs = [
        [[1, 0], [0, 1]],   # s=0: B00 + B11
        [[1, 0], [0, 0]],   # s=1: B00
        [[0, 1], [0, -1]],  # s=2: B01 - B11
        [[-1, 0], [1, 0]],  # s=3: B10 - B00
        [[0, 0], [0, 1]],   # s=4: B11
        [[1, 1], [0, 0]],   # s=5: B00 + B01
        [[0, 0], [1, 1]],   # s=6: B10 + B11
    ]

    c_coeffs = [
        [[1, 0], [0, 1]],   # s=0: C00 + C11
        [[0, 0], [1, -1]],  # s=1: C10 - C11
        [[0, 1], [0, 1]],   # s=2: C01 + C11
        [[1, 0], [1, 0]],   # s=3: C00 + C10
        [[-1, 1], [0, 0]],  # s=4: -C00 + C01
        [[0, 0], [0, 1]],   # s=5: C11
        [[1, 0], [0, 0]],   # s=6: C00
    ]

    # Build U, V, W using the tensor product structure:
    # For 4x4 = (2x2 blocks of 2x2), the decomposition is:
    #   U[i*4+j, 7*s+t] = a_coeffs[s][S][T] * a_coeffs[t][i2][j2]
    #     where i = 2*S + i2, j = 2*T + j2
    #   V[j*4+k, 7*s+t] = b_coeffs[s][T][K] * b_coeffs[t][j2][k2]
    #     where j = 2*T + j2, k = 2*K + k2
    #   W[k*4+i, 7*s+t] = c_coeffs[s][S][K] * c_coeffs[t][i2][k2]
    #     where i = 2*S + i2, k = 2*K + k2

    for s in range(7):
        for t in range(7):
            c_idx = 7 * s + t

            # U: row index = i*4 + j, where i = 2*S + i2, j = 2*T + j2
            for S in range(2):
                for T in range(2):
                    outer_a = a_coeffs[s][S][T]
                    if outer_a == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            inner_a = a_coeffs[t][i2][j2]
                            if inner_a == 0:
                                continue
                            i = 2 * S + i2
                            j = 2 * T + j2
                            row_u = i * 4 + j
                            U[row_u][c_idx] += outer_a * inner_a

            # V: row index = j*4 + k, where j = 2*T + j2, k = 2*K + k2
            for T in range(2):
                for K in range(2):
                    outer_b = b_coeffs[s][T][K]
                    if outer_b == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            inner_b = b_coeffs[t][j2][k2]
                            if inner_b == 0:
                                continue
                            j = 2 * T + j2
                            k = 2 * K + k2
                            row_v = j * 4 + k
                            V[row_v][c_idx] += outer_b * inner_b

            # W: row index = k*4 + i, where k = 2*K + k2, i = 2*S + i2
            for K in range(2):
                for S in range(2):
                    outer_c = c_coeffs[s][S][K]
                    if outer_c == 0:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            inner_c = c_coeffs[t][i2][k2]
                            if inner_c == 0:
                                continue
                            k = 2 * K + k2
                            i = 2 * S + i2
                            row_w = k * 4 + i
                            W[row_w][c_idx] += outer_c * inner_c

            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 15.0  # Safety margin well under 120s

    n, m, p = DIMENSIONS
    rank = 49  # 7 * 7: outer Strassen (7 terms) x inner Strassen (7 terms)

    # Strassen 2x2 coefficient matrices (7 terms each)
    a_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0: A00 + A11
        [[0, 0], [1, 1]],   # s=1: A10 + A11
        [[1, 0], [0, 0]],   # s=2: A00
        [[0, 0], [0, 1]],   # s=3: A11
        [[1, 1], [0, 0]],   # s=4: A00 + A01
        [[-1, 0], [1, 0]],  # s=5: A10 - A00
        [[0, 1], [0, -1]],  # s=6: A01 - A11
    ], dtype=float)

    b_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0: B00 + B11
        [[1, 0], [0, 0]],   # s=1: B00
        [[0, 1], [0, -1]],  # s=2: B01 - B11
        [[-1, 0], [1, 0]],  # s=3: B10 - B00
        [[0, 0], [0, 1]],   # s=4: B11
        [[1, 1], [0, 0]],   # s=5: B00 + B01
        [[0, 0], [1, 1]],   # s=6: B10 + B11
    ], dtype=float)

    c_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0: C00 + C11
        [[0, 0], [1, -1]],  # s=1: C10 - C11
        [[0, 1], [0, 1]],   # s=2: C01 + C11
        [[1, 0], [1, 0]],   # s=3: C00 + C10
        [[-1, 1], [0, 0]],  # s=4: -C00 + C01
        [[0, 0], [0, 1]],   # s=5: C11
        [[1, 0], [0, 0]],   # s=6: C00
    ], dtype=float)

    # Build U, V, W using Kronecker product structure
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for s in range(7):
        for t in range(7):
            c_idx = 7 * s + t
            U[:, c_idx] = np.kron(a_coeffs[s], a_coeffs[t]).flatten()
            V[:, c_idx] = np.kron(b_coeffs[s], b_coeffs[t]).flatten()
            W[:, c_idx] = np.kron(c_coeffs[s], c_coeffs[t]).flatten()

    # Convert to lists of lists with integer entries
    U_list = [[int(round(U[i, j])) for j in range(rank)] for i in range(n * m)]
    V_list = [[int(round(V[i, j])) for j in range(rank)] for i in range(m * p)]
    W_list = [[int(round(W[i, j])) for j in range(rank)] for i in range(p * n)]

    # Quick verification with one test case
    if time.time() < deadline - 2.0:
        A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        B = np.array([[16, 15, 14, 13], [12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]])
        C_expected = A @ B
        C_comp = np.zeros((n, p), dtype=float)
        U_np = np.array(U_list)
        V_np = np.array(V_list)
        W_np = np.array(W_list)
        for r in range(rank):
            u_r = U_np[:, r].reshape(n, m)
            v_r = V_np[:, r].reshape(m, p)
            m_r = u_r @ v_r
            w_r = W_np[:, r]
            for i in range(n):
                for k in range(p):
                    C_comp[i, k] += m_r[i, k] * w_r[k * n + i]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
