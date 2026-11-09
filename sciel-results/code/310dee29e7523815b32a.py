"""Matrix multiplication 3x4x8: rank-minimal exact tensor decomposition."""
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank = n * m * p  # 96
    
    # Create indices
    indices = list(product(range(n), range(m), range(p)))
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (i, j, k) in enumerate(indices):
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
