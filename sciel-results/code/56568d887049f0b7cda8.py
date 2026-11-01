"""Matrix multiplication 4x4x4 tensor decomposition."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np


def construct():
    """Return rank-49 Strassen-based decomposition for 4x4x4 matrix multiplication."""
    n, m, p = DIMENSIONS  # 4, 4, 4

    # 2x2 Strassen decomposition factors
    # U2[i*2+j, r]: A factor
    U2 = np.array([
        [1,  0,  1,  0,  1, -1,  0],
        [0,  0,  0,  0,  1,  0, -1],
        [0,  1,  0,  0,  0,  1,  0],
        [1,  1,  0,  1,  0,  0,  1],
    ], dtype=float)

    # V2[j*2+k, r]: B factor
    V2 = np.array([
        [1,  1,  0, -1,  0,  1,  0],
        [0,  0, -1,  0,  0,  1,  1],
        [0,  0,  0,  1,  0,  1,  1],
        [1,  0,  1,  0,  1,  0,  0],
    ], dtype=float)

    # W2[k*2+i, r]: C factor (k=col, i=row)
    W2 = np.array([
        [1,  0,  0,  1, -1,  0,  1],
        [0,  1,  0,  1,  0,  0,  0],
        [0,  0, 1,  0,  1,  0,  0],
        [1, -1, 1,  0,  0,  1,  0],
    ], dtype=float)

    # Build 4x4 via Kronecker product
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for i in range(n):
        for j in range(m):
            i1, i2 = i // 2, i % 2
            j1, j2 = j // 2, j % 2
            for r1 in range(7):
                for r2 in range(7):
                    r = r1 * 7 + r2
                    U[i * m + j, r] = U2[i1 * 2 + i2, r1] * U2[j1 * 2 + j2, r2]

    for j in range(m):
        for k in range(p):
            j1, j2 = j // 2, j % 2
            k1, k2 = k // 2, k % 2
            for r1 in range(7):
                for r2 in range(7):
                    r = r1 * 7 + r2
                    V[j * p + k, r] = V2[j1 * 2 + j2, r1] * V2[k1 * 2 + k2, r2]

    for k in range(p):
        for i in range(n):
            k1, k2 = k // 2, k % 2
            i1, i2 = i // 2, i % 2
            for r1 in range(7):
                for r2 in range(7):
                    r = r1 * 7 + r2
                    W[k * n + i, r] = W2[k1 * 2 + k2, r1] * W2[i1 * 2 + i2, r2]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

