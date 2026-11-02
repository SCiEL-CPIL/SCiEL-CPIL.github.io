import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = DIMENSIONS
    rank = 46

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_222(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return False

        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)

        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)

        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)

        U[a][col] += 1; U[d][col] += 1
        V[e][col] += 1; V[h][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1

        U[c][col] += 1; U[d][col] += 1
        V[e][col] += 1
        W[c10][col] += 1; W[c11][col] -= 1
        col += 1

        U[a][col] += 1
        V[f][col] += 1; V[h][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1

        U[d][col] += 1
        V[g][col] += 1; V[e][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1

        U[a][col] += 1; U[b][col] += 1
        V[h][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1

        U[c][col] += 1; U[a][col] -= 1
        V[e][col] += 1; V[f][col] += 1
        W[c11][col] += 1
        col += 1

        U[b][col] += 1; U[d][col] -= 1
        V[g][col] += 1; V[h][col] += 1
        W[c00][col] += 1
        col += 1

        return True

    def add_223(i0, j0, k0):
        nonlocal col
        if col + 9 > rank:
            return False
        add_strassen_222(i0, j0, k0)
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e2 = j0 * p + (k0 + 2)
        g2 = (j0 + 1) * p + (k0 + 2)
        c02 = (k0 + 2) * n + i0
        c12 = (k0 + 2) * n + (i0 + 1)

        U[a][col] += 1; V[e2][col] += 1; W[c02][col] += 1; col += 1
        U[b][col] += 1; V[g2][col] += 1; W[c02][col] += 1; col += 1
        U[c][col] += 1; V[e2][col] += 1; W[c12][col] += 1; col += 1
        U[d][col] += 1; V[g2][col] += 1; W[c12][col] += 1; col += 1

        return True

    # Strip 1: A cols 0,1 with B rows 0,1
    add_223(0, 0, 0)
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    add_strassen_222(0, 0, 3)
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    add_strassen_222(0, 0, 5)
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Strip 2: A cols 2,3 with B rows 2,3
    add_223(0, 2, 0)
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    add_strassen_222(0, 2, 3)
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    add_strassen_222(0, 2, 5)

    return {"U": U, "V": V, "W": W}
