"""Matrix multiplication 3,5,6 rank-minimal exact tensor decomposition."""
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _schoolbook():
    n, m, p = 3, 5, 6
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _to_list(x):
    if isinstance(x, dict):
        return {k: _to_list(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_list(v) for v in x]
    if isinstance(x, (int, float)):
        # Ensure integer values are represented as int for cleaner output
        if isinstance(x, float) and x == int(x):
            return int(x)
        return x
    return x

def construct():
    # The schoolbook decomposition has rank 90.
    # For 3x5x6, the trivial lower bound is max(3*5, 5*6, 6*3) = 30.
    # Without a specific known lower-rank exact decomposition for these dimensions
    # with the required entry types, we return the schoolbook decomposition.
    # This is a valid, exact decomposition.
    return _to_list(_schoolbook())

# EVOLVE-BLOCK-END
