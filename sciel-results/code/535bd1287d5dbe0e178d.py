"""Rank-minimal exact tensor decomposition for 2x4x6 matrix multiplication."""
import time
from fractions import Fraction as F

DIMENSIONS = [2, 4, 6]
_START_TIME = time.time()
DEADLINE = 150.0
MARGIN = 10.0

def _time_ok():
    return (time.time() - _START_TIME) < (DEADLINE - MARGIN)

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the schoolbook (rank-48) decomposition as (U, V, W) lists of lists of Fractions."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[F(0)] * rank for _ in range(n * m)]
    V = [[F(0)] * rank for _ in range(m * p)]
    W = [[F(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = F(1)
                V[j * p + k][col] = F(1)
                W[k * n + i][col] = F(1)
                col += 1
    return U, V, W

def construct():
    # Return the schoolbook decomposition.
    U, V, W = _schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
