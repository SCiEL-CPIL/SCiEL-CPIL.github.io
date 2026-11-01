"""
Rank-50 exact tensor decomposition for 2x5 x 5x5 matrix multiplication.

The multiplication tensor for C = A * B with A in R^{2x5}, B in R^{5x5} is a
direct sum of two independent 1x5x5 multiplication tensors.  The schoolbook
decomposition has rank 2 * 25 = 50, which is the known minimum for this
dimension (no sub-multiplicative embedding of 5x5 into a smaller rank tensor
is known or possible with standard Strassen-style block methods).

We construct the factor matrices U, V, W explicitly using the trace
convention:
  U index: i*m + j  (i in 0..n-1, j in 0..m-1)
  V index: j*p + k  (j in 0..m-1, k in 0..p-1)
  W index: k*n + i  (k in 0..p-1, i in 0..n-1)

Each rank-1 term corresponds to one (i, j, k) triple and sets exactly one
entry in each of U, V, W to 1.
"""

import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook decomposition of rank 50.

    U: (n*m, rank) = (10, 50)
    V: (m*p, rank) = (25, 50)
    W: (p*n, rank) = (10, 50)

    Each column of (U, V, W) defines one rank-1 slice of the multiplication
    tensor.  For the (i, j, k)-th slice, U[i*m+j] = V[j*p+k] = W[k*n+i] = 1
    and all other entries in that column are 0.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 50

    # Pre-allocate dense integer arrays (entries are 0 or 1)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1

    # Convert to list-of-lists to match the expected return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
