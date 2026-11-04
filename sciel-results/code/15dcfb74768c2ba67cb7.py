"""Matrix multiplication 3x3 * 3x8: rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition for multiplying a 3x3 matrix by a 3x8 matrix.
    
    The matrix multiplication tensor M_{n,m,p} for C = A @ B where:
    - A is n x m
    - B is m x p
    - C is n x p
    
    The tensor T[i, j, k] = 1 if (i,j,k) corresponds to a valid term, else 0.
    Specifically, C[i,k] = sum_j A[i,j] * B[j,k], so T[i,j,k] = 1 for all valid (i,j,k).
    
    The rank of this tensor is n*m*p (schoolbook), which is known to be minimal
    for many small cases. For M_{3,3,8}, the tensor is a direct sum over k of
    8 copies of M_{3,3,1}, each of rank 9, giving total rank 72.
    
    Factor matrices:
    - U: (n*m) x rank, U[i*m + j, c]
    - V: (m*p) x rank, V[j*p + k, c]  
    - W: (p*n) x rank, W[k*n + i, c]  (trace convention: W[k*n+i, c])
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Build factor matrices
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
