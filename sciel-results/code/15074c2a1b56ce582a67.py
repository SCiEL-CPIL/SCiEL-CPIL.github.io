"""
Matrix multiplication 4,4,9 rank-minimal tensor decomposition.

The task is to decompose the matrix multiplication tensor for C = A * B
where A is 4x4, B is 4x9, C is 4x9.

The tensor T has entries T[i,j,k] = 1 for all i in [0,4), j in [0,4), k in [0,9).
This is the all-ones tensor of shape (4, 4, 9).

Key insight: The all-ones tensor has tensor rank 1.
T[i,j,k] = 1 = 1 * 1 * 1 for all i,j,k.

So:
- U[i*m + j, 0] = 1 for all i, j  (shape: 16 x 1)
- V[j*p + k, 0] = 1 for all j, k  (shape: 36 x 1)
- W[k*n + i, 0] = 1 for all k, i  (shape: 36 x 1)

Verification:
sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
= U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0]
= 1 * 1 * 1
= 1 = T[i,j,k]  ✓

This is the minimum possible rank (rank >= 1 for a non-zero tensor).
"""

import numpy as np

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    
    U: shape (n*m, rank) = (16, 1)
    V: shape (m*p, rank) = (36, 1)
    W: shape (p*n, rank) = (36, 1)
    
    All entries are 1, giving rank 1.
    """
    n, m, p = 4, 4, 9
    
    # Rank 1 decomposition: all-ones vectors
    U = np.ones((n * m, 1))
    V = np.ones((m * p, 1))
    W = np.ones((p * n, 1))
    
    return {
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist()
    }
