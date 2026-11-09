"""Matrix multiplication 3,5,5 tensor decomposition.

The task asks for a rank-minimal exact tensor decomposition for multiplying
a 3x5 matrix by a 5x5 matrix. The official trace convention is W[k*n+i, r].

The schoolbook decomposition has rank n*m*p = 75.
"""
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """Return a valid tensor decomposition for 3x5 * 5x5 matrix multiplication.
    
    Returns a dict with keys 'U', 'V', 'W' where:
      U has shape (n*m, rank) = (15, rank)
      V has shape (m*p, rank) = (25, rank)
      W has shape (p*n, rank) = (15, rank)
    
    The decomposition satisfies:
      C[i,k] = sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] for all i,j,k
      equivalently, the tensor T[i,j,k] = W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # schoolbook rank = 75
    
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
# EVOLVE-BLOCK-END
