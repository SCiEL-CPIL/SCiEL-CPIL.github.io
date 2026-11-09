"""Matrix multiplication 2,4,8 tensor decomposition - block Strassen rank 56"""
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # A is n x m, B is m x p, C is n x p

    # Rank 56 using Strassen for 2x2x2 blocks
    # 8 blocks: s in {0,1} for j-split, t in {0,1,2,3} for k-split
    # Each block uses 7 Strassen multiplications

    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Strassen coefficients for 2x2x2
    # M1: (a00+a11)(b00+b11) -> C00 +=, C11 +=
    # M2: (a10+a11)b00 -> C10 +=, C11 -=
    # M3: a00(b01-b11) -> C01 +=, C11 +=
    # M4: a11(b10-b00) -> C00 +=, C10 +=
    # M5: (a00+a01)b11 -> C00 -=, C01 +=
    # M6: (a10-a00)(b00+b01) -> C11 +=
    # M7: (a01-a11)(b10+b11) -> C00 +=

    su = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1},
    ]

    sv = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1},
    ]

    sw = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(0, 0): -1, (1, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1},
    ]

    for s in range(2):
        for t in range(4):
            for r in range(7):
                for (ii, jj), val in su[r].items():
                    U[ii * m + (2 * s + jj)][col] += val
                for (jj, kk), val in sv[r].items():
                    V[(2 * s + jj) * p + (2 * t + kk)][col] += val
                for (kk, ii), val in sw[r].items():
                    W[(2 * t + kk) * n + ii][col] += val
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
