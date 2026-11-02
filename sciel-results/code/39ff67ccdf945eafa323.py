"""Matrix multiplication 3x5 by 5x6 — rank-minimal exact tensor decomposition.

Strategy:
  Use schoolbook algorithm with rank n*m*p = 90.
  This is guaranteed correct and serves as a baseline.
"""

import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    """
    Returns {'U', 'V', 'W'} with integer entries that exactly decompose
    the 3x5 @ 5x6 matrix multiplication tensor.
    
    U: (n*m, rank) = (15, rank)
    V: (m*p, rank) = (30, rank)
    W: (p*n, rank) = (18, rank)
    
    Convention:
      U[i*5 + k, c] selects A[i, k]
      V[k*6 + j, c] selects B[k, j]
      W[j*3 + i, c] accumulates into C[i, j]
    """
    n, m, p = DIMENSIONS  # n=3, m=5, p=6
    rank = n * m * p  # 90
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i*m + k][col] = 1
                V[k*p + j][col] = 1
                W[j*n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
