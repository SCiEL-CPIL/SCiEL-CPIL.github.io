import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5

    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1
        U[i0 * m + j0][col] = 1
        U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + k0][col] = 1
        V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = 1
        W[(k0+1) * n + (i0+1)][col] = 1
        col += 1

        # M2
        U[(i0+1) * m + j0][col] = 1
        U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + (i0+1)][col] = 1
        W[(k0+1) * n + (i0+1)][col] = -1
        col += 1

        # M3
        U[i0 * m + j0][col] = 1
        V[j0 * p + (k0+1)][col] = 1
        V[(j0+1) * p + (k0+1)][col] = -1
        W[(k0+1) * n + i0][col] = 1
        W[(k0+1) * n + (i0+1)][col] = 1
        col += 1

        # M4
        U[(i0+1) * m + (j0+1)][col] = 1
        V[(j0+1) * p + k0][col] = 1
        V[j0 * p + k0][col] = -1
        W[k0 * n + i0][col] = 1
        W[k0 * n + (i0+1)][col] = 1
        col += 1

        # M5
        U[i0 * m + j0][col] = 1
        U[i0 * m + (j0+1)][col] = 1
        V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = -1
        W[(k0+1) * n + i0][col] = 1
        col += 1

        # M6
        U[(i0+1) * m + j0][col] = 1
        U[i0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1
        V[j0 * p + (k0+1)][col] = 1
        W[(k0+1) * n + (i0+1)][col] = 1
        col += 1

        # M7
        U[i0 * m + (j0+1)][col] = 1
        U[(i0+1) * m + (j0+1)][col] = -1
        V[(j0+1) * p + k0][col] = 1
        V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = 1
        col += 1

    # Block A: j=0 (2x1 by 1x5) -> rank 10
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Block B: j=1,2 (2x2 by 2x5) -> rank 18
    # Split k into {0,1} Strassen(7), {2} schoolbook(4), {3,4} Strassen(7)

    # k=0,1: Strassen
    add_strassen(0, 1, 0)

    # k=2: Schoolbook (2x2 by 2x1)
    for i in range(n):
        for j in (1, 2):
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1

    # k=3,4: Strassen
    add_strassen(0, 1, 3)

    assert col == rank, f"Column count mismatch: {col} != {rank}"

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
