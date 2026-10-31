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
