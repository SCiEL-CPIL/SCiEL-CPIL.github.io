"""Matrix multiplication 3x5x5: return schoolbook decomposition."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook tensor decomposition for 3x5x5 matrix multiplication.
    
    The multiplication tensor for C = A * B where A is 3x5, B is 5x5, C is 3x5
    has rank 75 using the standard schoolbook algorithm.
    
    U shape: (3*5, 75) = (15, 75)
    V shape: (5*5, 75) = (25, 75)
    W shape: (5*3, 75) = (15, 75) -- Note: The prompt says W[k*n+i, r].
    
    The prompt specifies:
    U: (n*m, rank) -> (15, rank)
    V: (m*p, rank) -> (25, rank)
    W: (p*n, rank) -> (15, rank)
    
    And the output factor follows W[k*n+i, r].
    This implies the index for W is $k \cdot n + i$.
    $k$ ranges $0..p-1$ (5), $i$ ranges $0..n-1$ (3).
    So $W$ has $p \cdot n = 15$ rows.
    
    The decomposition is:
    For each i in 0..n-1, j in 0..m-1, k in 0..p-1:
    col = i*m*p + j*p + k
    U[i*m + j, col] = 1
    V[j*p + k, col] = 1
    W[k*n + i, col] = 1
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
