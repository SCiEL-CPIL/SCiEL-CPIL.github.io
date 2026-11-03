"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total_deadline = 150.0
    start_time = time.time()
    
    # For 2x3x7 matrix multiplication, we use the schoolbook decomposition
    # which has rank n*m*p = 42. This is guaranteed to be correct.
    #
    # The tensor T for matrix multiplication C = A*B where A is n x m and B is m x p
    # has entries T[i,j,k] for i in range(n), j in range(m), k in range(p).
    # The decomposition expresses T as a sum of rank-1 tensors:
    #   T = sum_{r=0}^{rank-1} U[:,r] * V[:,r] * W[:,r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    #
    # The index conventions are:
    #   U index: i*m + j  (i-th row of A, j-th column of A / j-th row of B)
    #   V index: j*p + k  (j-th row of B, k-th column of B / k-th row of C)
    #   W index: k*n + i  (k-th row of C, i-th column of C)
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
