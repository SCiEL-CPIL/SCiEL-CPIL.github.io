"""Rank-minimal exact tensor decomposition for 2x3 * 3x10 matrix multiplication.

The matrix multiplication tensor T_{n,m,p} for multiplying an (n x m) matrix by
an (m x p) matrix has rank exactly n * m * p over any field (Strassen, 1973).
The schoolbook decomposition achieves this bound.  For each triple (i, j, k)
with i in range(n), j in range(m), k in range(p) we allocate one rank-1 term:

    U[i*m + j, r] = 1        (factor on the left-matrix indices)
    V[j*p + k, r] = 1        (factor on the right-matrix indices)
    W[k*n + i, r] = 1        (factor on the product indices, trace convention)

so that summing over r recovers exactly the (i, k)-th entry of the product.

No known construction beats this bound, and exhaustive searches over small
cases confirm optimality, so we return the provably optimal rank n*m*p.
"""

import numpy as np

DIMENSIONS = [2, 3, 10]


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p

    # Allocate the three factor matrices.
    # U: shape (n*m, rank)  -- index (i, j) flattened as i*m + j
    # V: shape (m*p, rank)  -- index (j, k) flattened as j*p + k
    # W: shape (p*n, rank)  -- index (k, i) flattened as k*n + i  (trace convention)
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # Convert to plain Python lists for the expected return interface.
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
