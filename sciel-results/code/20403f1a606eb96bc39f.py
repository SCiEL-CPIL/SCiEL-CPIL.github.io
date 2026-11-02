import numpy as np
import time

# Entry point required by the evaluator
def construct():
    """
    Build a rank-56 exact tensor decomposition for multiplying a 2x4 matrix by a 4x8 matrix.
    Uses Strassen's algorithm on 8 disjoint 2x2x2 sub-problems.
    Returns a dict with keys 'U', 'V', 'W' containing integer numpy arrays.
    """
    return _construct_rank_56()


def _construct_rank_56():
    n, m, p = 2, 4, 8
    R = 56

    U = np.zeros((n * m, R), dtype=np.int8)
    V = np.zeros((m * p, R), dtype=np.int8)
    W = np.zeros((p * n, R), dtype=np.int8)

    for i in range(n):
        for j in range(m // 2):
            for k in range(p // 2):
                r0 = 7 * (i * (m // 2) + j * (p // 2) + k)

                a11 = i * m + 2 * j
                a12 = i * m + 2 * j + 1
                a21 = (i + 1) * m + 2 * j
                a22 = (i + 1) * m + 2 * j + 1

                b11 = 2 * j * p + 2 * k
                b12 = 2 * j * p + 2 * k + 1
                b21 = (2 * j + 1) * p + 2 * k
                b22 = (2 * j + 1) * p + 2 * k + 1

                c11 = 2 * k * n + i
                c12 = (2 * k + 1) * n + i
                c21 = 2 * k * n + (i + 1)
                c22 = (2 * k + 1) * n + (i + 1)

                # M1 = (a11 + a22)(b11 + b22)
                U[a11, r0] += 1
                U[a22, r0] += 1
                V[b11, r0] += 1
                V[b22, r0] += 1
                W[c11, r0] += 1
                W[c22, r0] += 1

                # M2 = (a21 + a22) b11
                U[a21, r0 + 1] += 1
                U[a22, r0 + 1] += 1
                V[b11, r0 + 1] += 1
                W[c21, r0 + 1] += 1
                W[c22, r0 + 1] -= 1

                # M3 = a11 (b12 - b22)
                U[a11, r0 + 2] += 1
                V[b12, r0 + 2] += 1
                V[b22, r0 + 2] -= 1
                W[c12, r0 + 2] += 1
                W[c22, r0 + 2] += 1

                # M4 = a22 (b21 - b11)
                U[a22, r0 + 3] += 1
                V[b21, r0 + 3] += 1
                V[b11, r0 + 3] -= 1
                W[c11, r0 + 3] += 1
                W[c21, r0 + 3] += 1

                # M5 = (a11 + a12) b22
                U[a11, r0 + 4] += 1
                U[a12, r0 + 4] += 1
                V[b22, r0 + 4] += 1
                W[c11, r0 + 4] -= 1
                W[c12, r0 + 4] += 1

                # M6 = (a21 - a11)(b11 + b12)
                U[a21, r0 + 5] += 1
                U[a11, r0 + 5] -= 1
                V[b11, r0 + 5] += 1
                V[b12, r0 + 5] += 1
                W[c22, r0 + 5] += 1

                # M7 = (a12 - a22)(b21 + b22)
                U[a12, r0 + 6] += 1
                U[a22, r0 + 6] -= 1
                V[b21, r0 + 6] += 1
                V[b22, r0 + 6] += 1
                W[c11, r0 + 6] += 1

    return {"U": U, "V": V, "W": W}
