import time
import numpy as np

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0

def construct():
    n, m, p = DIMENSIONS
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        U[i_base * m + j_base][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1

        U[(i_base + 1) * m + j_base][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] -= 1
        col += 1

        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] -= 1
        W[(k_base + 1) * n + i_base][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1

        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[(j_base + 1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        col += 1

        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base + 1)][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base + 1) * n + i_base][col] += 1
        col += 1

        U[(i_base + 1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1

        U[i_base * m + (j_base + 1)][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] -= 1
        V[(j_base + 1) * p + k_base][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    def add_naive_2112_block(i_base, j_base, k_base):
        nonlocal col
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        W[(k_base + 1) * n + i_base][col] += 1
        col += 1

        U[(i_base + 1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        col += 1

        U[(i_base + 1) * m + j_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1

    # Block 1: A cols 0,1 (j=0,1). B rows 0,1 (j=0,1).
    # Pairs of k: (0,1), (2,3), (4,5).
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)

    # Block 2: A cols 2,3 (j=2,3). B rows 2,3 (j=2,3).
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)

    # Block 3: A col 4 (j=4). B row 4 (j=4).
    add_naive_2112_block(0, 4, 0)
    add_naive_2112_block(0, 4, 2)
    add_naive_2112_block(0, 4, 4)

    # Verify col == 54
    assert col == 54, f"Expected rank 54 but got {col}"

    return {"U": U, "V": V, "W": W}
