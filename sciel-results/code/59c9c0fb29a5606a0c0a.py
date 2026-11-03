"""Matrix multiplication 2x3x8 with rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen_2x2x2(i_off, j_off, k_off):
        nonlocal col
        # 1
        U[(i_off) * m + (j_off)][col] += 1
        U[(i_off + 1) * m + (j_off + 1)][col] += 1
        V[(j_off) * p + (k_off)][col] += 1
        V[(j_off + 1) * p + (k_off + 1)][col] += 1
        W[(k_off) * n + (i_off)][col] += 1
        W[(k_off + 1) * n + (i_off + 1)][col] += 1
        col += 1

        # 2
        U[(i_off + 1) * m + (j_off)][col] += 1
        U[(i_off + 1) * m + (j_off + 1)][col] += 1
        V[(j_off) * p + (k_off)][col] += 1
        W[(k_off) * n + (i_off + 1)][col] += 1
        W[(k_off + 1) * n + (i_off + 1)][col] += -1
        col += 1

        # 3
        U[(i_off) * m + (j_off)][col] += 1
        V[(j_off) * p + (k_off + 1)][col] += 1
        V[(j_off + 1) * p + (k_off + 1)][col] += -1
        W[(k_off + 1) * n + (i_off)][col] += 1
        W[(k_off + 1) * n + (i_off + 1)][col] += 1
        col += 1

        # 4
        U[(i_off + 1) * m + (j_off + 1)][col] += 1
        V[(j_off + 1) * p + (k_off)][col] += 1
        V[(j_off) * p + (k_off)][col] += -1
        W[(k_off) * n + (i_off)][col] += 1
        W[(k_off) * n + (i_off + 1)][col] += 1
        col += 1

        # 5
        U[(i_off) * m + (j_off)][col] += 1
        U[(i_off) * m + (j_off + 1)][col] += 1
        V[(j_off + 1) * p + (k_off + 1)][col] += 1
        W[(k_off) * n + (i_off)][col] += -1
        W[(k_off + 1) * n + (i_off)][col] += 1
        col += 1

        # 6
        U[(i_off + 1) * m + (j_off)][col] += 1
        U[(i_off) * m + (j_off)][col] += -1
        V[(j_off) * p + (k_off)][col] += 1
        V[(j_off) * p + (k_off + 1)][col] += 1
        W[(k_off + 1) * n + (i_off + 1)][col] += 1
        col += 1

        # 7
        U[(i_off) * m + (j_off + 1)][col] += 1
        U[(i_off + 1) * m + (j_off + 1)][col] += -1
        V[(j_off + 1) * p + (k_off)][col] += 1
        V[(j_off + 1) * p + (k_off + 1)][col] += 1
        W[(k_off) * n + (i_off)][col] += 1
        col += 1

    def add_schoolbook_2x1x4(i_off, j_off, k_off):
        nonlocal col
        for i in range(2):
            for k in range(4):
                U[(i_off + i) * m + (j_off)][col] = 1
                V[(j_off) * p + (k_off + k)][col] = 1
                W[(k_off + k) * n + (i_off + i)][col] = 1
                col += 1

    # First p=4 block
    add_strassen_2x2x2(0, 0, 0)
    add_strassen_2x2x2(0, 0, 2)
    add_schoolbook_2x1x4(0, 2, 0)

    # Second p=4 block
    add_strassen_2x2x2(0, 0, 4)
    add_strassen_2x2x2(0, 0, 6)
    add_schoolbook_2x1x4(0, 2, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
