import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 4
    rank = 49

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen's 7 multiplications for 2x2
    # a_coeffs[s][i][j]: coefficient of A[i][j] in the s-th Strassen A-form
    # b_coeffs[s][j][k]: coefficient of B[j][k] in the s-th Strassen B-form
    # c_coeffs[s][i][k]: coefficient for C[i][k] from the s-th Strassen term

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
        [[1, 0], [0, 1]],   # s=0: C00 += M, C11 += M
        [[0, 0], [1, -1]],  # s=1: C10 += M, C11 -= M
        [[0, 1], [0, 1]],   # s=2: C01 += M, C11 += M
        [[1, 0], [1, 0]],   # s=3: C00 += M, C10 += M
        [[-1, 1], [0, 0]],  # s=4: C00 -= M, C01 += M
        [[0, 0], [0, 1]],   # s=5: C11 += M
        [[1, 0], [0, 0]],   # s=6: C00 += M
    ]

    # Build 49-rank decomposition via recursive Strassen
    # A[i,j], B[j,k], C[i,k] with i,j,k in 0..3
    # Split into 2x2 blocks: i = 2*S + s2, j = 2*T + t2, k = 2*K + k2
    # W follows trace convention: W[k*n+i, r] = W[k*4+i, r]

    for s in range(7):
        for t in range(7):
            c_idx = 7 * s + t

            # U[i*4+j, c_idx]: coefficient of A[i,j]
            for S in range(2):
                for T in range(2):
                    a_outer = a_coeffs[s][S][T]
                    if a_outer == 0:
                        continue
                    for s2 in range(2):
                        for t2 in range(2):
                            a_inner = a_coeffs[t][s2][t2]
                            if a_inner == 0:
                                continue
                            i = 2 * S + s2
                            j = 2 * T + t2
                            U[i * 4 + j][c_idx] += a_outer * a_inner

            # V[j*4+k, c_idx]: coefficient of B[j,k]
            for T in range(2):
                for K in range(2):
                    b_outer = b_coeffs[s][T][K]
                    if b_outer == 0:
                        continue
                    for t2 in range(2):
                        for k2 in range(2):
                            b_inner = b_coeffs[t][t2][k2]
                            if b_inner == 0:
                                continue
                            j = 2 * T + t2
                            k = 2 * K + k2
                            V[j * 4 + k][c_idx] += b_outer * b_inner

            # W[k*4+i, c_idx]: coefficient for C[i,k]
            for K in range(2):
                for S in range(2):
                    c_outer = c_coeffs[s][S][K]
                    if c_outer == 0:
                        continue
                    for k2 in range(2):
                        for s2 in range(2):
                            c_inner = c_coeffs[t][s2][k2]
                            if c_inner == 0:
                                continue
                            k = 2 * K + k2
                            i = 2 * S + s2
                            W[k * 4 + i][c_idx] += c_outer * c_inner

            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
