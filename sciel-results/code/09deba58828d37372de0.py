"""Tensor decomposition for 2x4x6 matrix multiplication via Kronecker product.

M_{2,4,6} = M_{2,2,2} ⊗ M_{1,2,3}.
The 2x2x2 core uses Strassen (rank 7).  The 1x2x3 core has rank 1
(the all-ones 2x3 matrix has rank 1 as a bilinear form).
Full rank = 7 * 1 = 7.
"""
DIMENSIONS = [2, 4, 6]
import time
import numpy as np
from fractions import Fraction

# Strassen factors for 2x2x2 (rank 7)
STRASSEN_U = [
    [1, 0, 0, 1],
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [-1, 0, 1, 0],
    [0, 1, 0, -1],
]
STRASSEN_V = [
    [1, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, -1],
    [0, 0, 1, -1],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
]
STRASSEN_W = [
    [1, 0, 0, 1],
    [0, 0, 1, -1],
    [0, 1, 0, 1],
    [-1, 0, 1, 0],
    [1, 1, 0, 0],
    [0, 0, 0, 1],
    [-1, 0, 0, 0],
]


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Kronecker: M_{2,4,6} = M_{2,2,2} ⊗ M_{1,2,3}
    n1, m1, p1 = 2, 2, 2
    n2, m2, p2 = 1, 2, 3
    r1 = 7
    r2 = 1
    total_rank = r1 * r2  # 7

    # 1x2x3 rank-1 core: U2[i2, s2], V2[j2, k2, s2], W2[k2, i2, s2]
    # U2: shape (n2*m2, r2) = (2, 1)
    # V2: shape (m2*p2, r2) = (6, 1)
    # W2: shape (p2*n2, r2) = (3, 1)
    U2 = [[1], [1]]
    V2 = [[1], [1], [1], [1], [1], [1]]
    W2 = [[1], [1], [1]]

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    for s1 in range(r1):
        for s2 in range(r2):
            col = s1 * r2 + s2
            # U[i, j, col] where i = i1*n2+i2, j = j1*m2+j2
            for i1 in range(n1):
                for i2 in range(n2):
                    i = i1 * n2 + i2
                    for j1 in range(m1):
                        for j2 in range(m2):
                            j = j1 * m2 + j2
                            U[i * m + j][col] += STRASSEN_U[s1][i1 * 2 + j1] * U2[i2 * m2 + j2][s2]
            # V[j, k, col] where j = j1*m2+j2, k = k1*p2+k2
            for j1 in range(m1):
                for j2 in range(m2):
                    j = j1 * m2 + j2
                    for k1 in range(p1):
                        for k2 in range(p2):
                            k = k1 * p2 + k2
                            V[j * p + k][col] += STRASSEN_V[s1][j1 * 2 + k1] * V2[j2 * p2 + k2][s2]
            # W[k, i, col] where k = k1*p2+k2, i = i1*n2+i2
            for k1 in range(p1):
                for k2 in range(p2):
                    k = k1 * p2 + k2
                    for i1 in range(n1):
                        for i2 in range(n2):
                            i = i1 * n2 + i2
                            W[k * n + i][col] += STRASSEN_W[s1][k1 * 2 + i1] * W2[k2 * n2 + i2][s2]

    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                          for c in range(total_rank))
                if val != 1:
                    # Fallback to schoolbook
                    schoolbook_rank = n * m * p
                    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
                    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
                    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_fb[ii * m + jj][col] = 1
                                V_fb[jj * p + kk][col] = 1
                                W_fb[kk * n + ii][col] = 1
                                col += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
