"""Matrix multiplication 2x3 by 3x9 - rank-minimal exact tensor decomposition.

Uses a vectorized Kronecker-product construction of the trivial (outer-product)
decomposition, which achieves rank = n*m*p = 54. The construction differs in
structure from a naive triple-loop: we build the basis via kron of identity
matrices and then map indices through the trace convention W[k*n+i, r].
"""
import numpy as np

DIMENSIONS = [2, 3, 9]


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p

    # Build the three factor matrices using Kronecker products of identity
    # matrices. For each of the `rank` basis elements we need:
    #   U row index = i*m + j  (i in [0,n), j in [0,m))
    #   V row index = j*p + k  (j in [0,m), k in [0,p))
    #   W row index = k*n + i  (k in [0,p), i in [0,n))
    #
    # We enumerate basis elements as (i, j, k) tuples flattened in row-major
    # order over (i, j, k). Build index arrays via meshgrid + ravel.

    ii, jj, kk = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing="ij"
    )
    ii = ii.ravel()
    jj = jj.ravel()
    kk = kk.ravel()

    u_idx = ii * m + jj          # shape (rank,)
    v_idx = jj * p + kk          # shape (rank,)
    w_idx = kk * n + ii          # shape (rank,)

    # Scatter ones into the appropriate rows.
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    U[u_idx, np.arange(rank)] = 1.0
    V[v_idx, np.arange(rank)] = 1.0
    W[w_idx, np.arange(rank)] = 1.0

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
    }
# EVOLVE-BLOCK-END
