import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # The target tensor is the all-ones tensor T[i, j, k] = 1.
    # This tensor has rank 1.
    # A rank-1 decomposition is U[i, j] = 1, V[j, k] = 1, W[k, i] = 1.
    
    U = [[1 for _ in range(1)] for _ in range(n * m)]
    V = [[1 for _ in range(1)] for _ in range(m * p)]
    W = [[1 for _ in range(1)] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
