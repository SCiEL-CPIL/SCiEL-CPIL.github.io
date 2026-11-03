"""
Rank-36 exact tensor decomposition for 2x4 by 4x5 matrix multiplication.
Uses Strassen for 2x2x2 blocks and schoolbook for 2x2x1 blocks.
"""

import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140

def construct():
    n, m, p = 2, 4, 5
    rank = 36

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_222(i0, j0, k0, col_start):
        nonlocal col
        i0i = i0
        i1i = i0 + 1
        j0i = j0
        j1i = j0 + 1
        k0i = k0
        k1i = k0 + 1

        # M1 = (A00 + A11) * (B00 + B11)
        c = col_start
        U[i0i * m + j0i][c] += 1
        U[i1i * m + j1i][c] += 1
        V[j0i * p + k0i][c] += 1
        V[j1i * p + k1i][c] += 1
        W[k0i * n + i0i][c] += 1
        W[k1i * n + i1i][c] += 1
        col += 1

        # M2 = (A10 + A11) * B00
        c = col_start + 1
        U[i1i * m + j0i][c] += 1
        U[i1i * m + j1i][c] += 1
        V[j0i * p + k0i][c] += 1
        W[k0i * n + i1i][c] += 1
        W[k1i * n + i1i][c] -= 1
        col += 1

        # M3 = A00 * (B01 - B11)
        c = col_start + 2
        U[i0i * m + j0i][c] += 1
        V[j0i * p + k1i][c] += 1
        V[j1i * p + k1i][c] -= 1
        W[k1i * n + i0i][c] += 1
        W[k1i * n + i1i][c] += 1
        col += 1

        # M4 = A11 * (B10 - B00)
        c = col_start + 3
        U[i1i * m + j1i][c] += 1
        V[j1i * p + k0i][c] += 1
        V[j0i * p + k0i][c] -= 1
        W[k0i * n + i0i][c] += 1
        W[k0i * n + i1i][c] += 1
        col += 1

        # M5 = (A00 + A01) * B11
        c = col_start + 4
        U[i0i * m + j0i][c] += 1
        U[i0i * m + j1i][c] += 1
        V[j1i * p + k1i][c] += 1
        W[k0i * n + i0i][c] -= 1
        W[k1i * n + i0i][c] += 1
        col += 1

        # M6 = (A10 - A00) * (B00 + B01)
        c = col_start + 5
        U[i1i * m + j0i][c] += 1
        U[i0i * m + j0i][c] -= 1
        V[j0i * p + k0i][c] += 1
        V[j0i * p + k1i][c] += 1
        W[k1i * n + i1i][c] += 1
        col += 1

        # M7 = (A01 - A11) * (B10 + B11)
        c = col_start + 6
        U[i0i * m + j1i][c] += 1
        U[i1i * m + j1i][c] -= 1
        V[j1i * p + k0i][c] += 1
        V[j1i * p + k1i][c] += 1
        W[k0i * n + i0i][c] += 1
        col += 1

        return col_start + 7

    def add_schoolbook_2x2x1(i0, j0, k0, col_start):
        c = col_start
        for di in range(2):
            for dj in range(2):
                U[(i0 + di) * m + (j0 + dj)][c] = 1
                V[(j0 + dj) * p + k0][c] = 1
                W[k0 * n + (i0 + di)][c] = 1
                c += 1
        return c

    # Four Strassen blocks for the 2x4 by 4x4 part
    col = add_strassen_222(0, 0, 0, col)
    col = add_strassen_222(0, 2, 0, col)
    col = add_strassen_222(0, 0, 2, col)
    col = add_strassen_222(0, 2, 2, col)

    # Two schoolbook blocks for the 2x4 by 4x1 part
    col = add_schoolbook_2x2x1(0, 0, 4, col)
    col = add_schoolbook_2x2x1(0, 2, 4, col)

    assert col == rank, f"Expected rank {rank}, got {col}"

    if time.time() > _DEADLINE:
        raise TimeoutError("Deadline exceeded")

    return {"U": U, "V": V, "W": W}
