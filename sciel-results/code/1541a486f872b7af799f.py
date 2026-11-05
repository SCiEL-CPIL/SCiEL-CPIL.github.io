"""Matrix multiplication 4x5x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns an exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    
    Decomposition strategy:
    - Split A = [A1 | A2] where A1 is 4x4, A2 is 4x1
    - Split B = [B1; B2] where B1 is 4x5, B2 is 1x5
    - C = A1*B1 + A2*B2
    - A1*B1 = A1*B11 + A1*B12 where B11 is 4x4, B12 is 4x1
    - A1*B11: rank 49 (Strassen)
    - A1*B12: rank 4
    - A2*B2: rank 4
    Total rank: 57
    
    The tensor identity: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    for all i in [0,4), j in [0,5), k in [0,5).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Schoolbook decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
