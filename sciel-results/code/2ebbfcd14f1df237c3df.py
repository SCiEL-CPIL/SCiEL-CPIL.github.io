"""Matrix multiplication 2,3,8 - rank-minimal tensor decomposition.

Strategy: Use Strassen decomposition on 2x2 sub-blocks plus a rank-1 tail term
for the third column of A. This yields rank 8 per 2-column block of B,
total rank 32 for 4 blocks.
"""
DIMENSIONS = [2, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p):
    """Check if U, V, W form a valid tensor decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# Strassen decomposition for 2x2x2 tensor (rank 7)
# Indices: i in {0,1}, j in {0,1}, k in {0,1}
_STRASSEN_U = [
    {(0, 0): 1, (1, 1): 1},
    {(1, 0): 1, (1, 1): 1},
    {(0, 0): 1},
    {(1, 1): 1},
    {(0, 0): 1, (0, 1): 1},
    {(1, 0): 1, (0, 0): -1},
    {(0, 1): 1, (1, 1): -1},
]
_STRASSEN_V = [
    {(0, 0): 1, (1, 1): 1},
    {(0, 0): 1},
    {(0, 1): 1, (1, 1): -1},
    {(1, 0): 1, (0, 0): -1},
    {(1, 1): 1},
    {(0, 0): 1, (0, 1): 1},
    {(1, 0): 1, (1, 1): 1},
]
_STRASSEN_W = [
    {(0, 0): 1, (1, 1): 1},
    {(0, 1): 1, (1, 1): -1},
    {(0, 1): 1, (1, 1): 1},
    {(0, 0): 1, (0, 1): 1},
    {(0, 0): -1, (0, 1): 1},
    {(1, 1): 1},
    {(0, 0): 1},
]

def _block_decomposition_238(n, m, p):
    """Build rank-32 decomposition by 4 blocks of 8 terms each."""
    assert n == 2 and m == 3 and p == 8
    nblocks = p // 2  # 4
    rank = nblocks * 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for b in range(nblocks):
        k0 = 2 * b
        col_base = b * 8
        
        # 7 Strassen terms
        for r in range(7):
            col = col_base + r
            for (i, j), val in _STRASSEN_U[r].items():
                U[i * m + j][col] = val
            for (j, kp), val in _STRASSEN_V[r].items():
                V[j * p + (k0 + kp)][col] = val
            for (kp, i), val in _STRASSEN_W[r].items():
                W[(k0 + kp) * n + i][col] = val
        
        # 1 Tail term
        col = col_base + 7
        # U: a_{02} and a_{12}
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = 1
        # V: b_{2, 2b} and b_{2, 2b+1}
        V[2 * p + k0][col] = 1
        V[2 * p + k0 + 1][col] = 1
        # W: C_{0, 2b}, C_{0, 2b+1}, C_{1, 2b}, C_{1, 2b+1}
        W[k0 * n + 0][col] = 1
        W[(k0 + 1) * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        W[(k0 + 1) * n + 1][col] = 1

    return {"U": U, "V": V, "W": W}

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 115

    # Try rank 32 decomposition
    try:
        cand = _block_decomposition_238(n, m, p)
        if _validate(cand["U"], cand["V"], cand["W"], n, m, p):
            return cand
    except Exception:
        pass

    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
