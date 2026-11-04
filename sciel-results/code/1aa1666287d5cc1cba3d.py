import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 120.0
_MARGIN = 5.0


def _time_ok():
    return time.time() < _DEADLINE - _MARGIN


def construct():
    n, m, p = DIMENSIONS

    # Strassen 2x2 factors (rank 7)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=np.int8)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=np.int8)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=np.int8)

    # Build 4x4 factors via Kronecker structure: rank = 7*7 = 49
    R = 49
    U = np.zeros((n * m, R), dtype=np.int8)
    V = np.zeros((m * p, R), dtype=np.int8)
    W = np.zeros((p * n, R), dtype=np.int8)

    if not _time_ok():
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Precompute all outer products for U2 (4x7 -> 7x7 each, 4 of them)
    U2_outers = [np.outer(U2[i], U2[j]) for i in range(4) for j in range(4)]
    # Reorganize: U2_outers[(i1*2+i2) * 4 + (j1*2+j2)] is outer of U2[i1*2+i2], U2[j1*2+j2]
    # Actually let's just index by (i, j) pairs where i,j are in 0..3
    # U[i*m+j] for i in 0..3, j in 0..3
    # i = i1*2 + i2, j = j1*2 + j2
    # U2 row for first component: i1*2+j1, second component: i2*2+j2
    # So U[i*m+j] = outer(U2[i1*2+j1], U2[i2*2+j2])

    for i in range(4):
        i1, i2 = i >> 1, i & 1
        for j in range(4):
            j1, j2 = j >> 1, j & 1
            idx1 = i1 * 2 + j1
            idx2 = i2 * 2 + j2
            U[i * 4 + j] = (U2[idx1][:, None] * U2[idx2][None, :]).ravel()

    if not _time_ok():
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    for j in range(4):
        j1, j2 = j >> 1, j & 1
        for k in range(4):
            k1, k2 = k >> 1, k & 1
            idx1 = j1 * 2 + k1
            idx2 = j2 * 2 + k2
            V[j * 4 + k] = (V2[idx1][:, None] * V2[idx2][None, :]).ravel()

    if not _time_ok():
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    for k in range(4):
        k1, k2 = k >> 1, k & 1
        for i in range(4):
            i1, i2 = i >> 1, i & 1
            idx1 = k1 * 2 + i1
            idx2 = k2 * 2 + i2
            W[k * 4 + i] = (W2[idx1][:, None] * W2[idx2][None, :]).ravel()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
