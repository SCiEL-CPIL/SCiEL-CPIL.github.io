"""Matrix multiplication 2x3x10 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct an exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x10 matrix.
    
    Strategy: One rank-1 tensor term per (j,k) pair where j is the shared
    dimension (3) and k is the output column (10). Each term handles all
    rows of A (2 rows) for a fixed B row j and output column k.
    
    Total rank = m * p = 3 * 10 = 30.
    
    W follows convention W[k*n + i, r] where k is output row (0..p-1), i is output col (0..n-1).
    """
    deadline = time.time() + 130.0
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    rank = m * p  # 30
    U = [[0] * rank for _ in range(n * m)]   # 6 x 30
    V = [[0] * rank for _ in range(m * p)]   # 30 x 30
    W = [[0] * rank for _ in range(p * n)]   # 20 x 30
    
    col = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                U[i * m + j][col] = 1
                W[k * n + i][col] = 1
            V[j * p + k][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
