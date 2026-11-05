"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 2, 3, 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    r = 0

    strassen_u = [
        [(0, 1), (4, 1)],
        [(3, 1), (4, 1)],
        [(0, 1)],
        [(4, 1)],
        [(0, 1), (1, 1)],
        [(3, 1), (0, -1)],
        [(1, 1), (4, -1)],
    ]

    strassen_v = [
        [(0, 1), (11, 1)],
        [(0, 1)],
        [(1, 1), (11, -1)],
        [(10, 1), (0, -1)],
        [(11, 1)],
        [(0, 1), (1, 1)],
        [(10, 1), (11, 1)],
    ]

    strassen_w = [
        [(0, 1), (3, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (3, 1)],
        [(0, 1), (1, 1)],
        [(0, -1), (2, 1)],
        [(3, 1)],
        [(0, 1)],
    ]

    v_map = {0: 0, 1: 1, 10: 10, 11: 11}

    for b in range(5):
        col0 = 2 * b
        col1 = 2 * b + 1

        for s in range(7):
            for (u_off, coeff) in strassen_u[s]:
                U[u_off][r] += coeff

            for (v_off, coeff) in strassen_v[s]:
                base = v_map[v_off]
                if base < 10:
                    v_row = base + 10 * b
                else:
                    v_row = base + 10 * b
                V[v_row][r] += coeff

            for (w_off, coeff) in strassen_w[s]:
                if w_off == 0:
                    w_row = col0 * 2 + 0
                elif w_off == 1:
                    w_row = col0 * 2 + 1
                elif w_off == 2:
                    w_row = col1 * 2 + 0
                elif w_off == 3:
                    w_row = col1 * 2 + 1
                W[w_row][r] += coeff
            r += 1

    for i in range(2):
        for k in range(10):
            U[i * 3 + 2][r] = 1
            V[20 + k][r] = 1
            W[k * 2 + i][r] = 1
            r += 1

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                if not np.isclose(lhs, 1.0, atol=1e-9):
                    rank = 60
                    U2 = [[0] * rank for _ in range(n * m)]
                    V2 = [[0] * rank for _ in range(m * p)]
                    W2 = [[0] * rank for _ in range(p * n)]
                    for rr in range(rank):
                        ii = rr // (m * p)
                        rem = rr % (m * p)
                        jj = rem // p
                        kk = rem % p
                        U2[ii * m + jj][rr] = 1
                        V2[jj * p + kk][rr] = 1
                        W2[kk * n + ii][rr] = 1
                    return {"U": U2, "V": V2, "W": W2}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
