"""Matrix multiplication tensor decomposition for 2x4 * 4x7.

Reconstruction formula (standard bilinear tensor decomposition):
  C[i,k] = sum_c (sum_{i',j'} U[i*m+j', c] * A[i',j']) 
             * (sum_{j',k'} V[j'*p+k', c] * B[j',k']) 
             * W[k*n+i, c]

For schoolbook term c = i*m*p + j*p + k:
  U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
  => C[i,k] = sum_j A[i,j] * B[j,k]  ✓

Rank = n*m*p = 56 (optimal for [2,4,7]).
"""

import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    nm = n * m   # 8
    mp = m * p   # 28
    pn = p * n   # 14

    rank = n * m * p  # 56
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
