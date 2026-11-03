"""Matrix multiplication tensor decomposition for 5x5 by 5x5."""
import itertools
import random
import sys
import time

import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START


def construct():
    n, m, p = 5, 5, 5
    rank = 125
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i, j, k in itertools.product(range(n), range(m), range(p)):
        c = i * m * p + j * p + k
        U[i * m + j][c] = 1
        V[j * p + k][c] = 1
        W[k * n + i][c] = 1
    return {"U": U, "V": V, "W": W}


# EVOLVE-BLOCK-END
