"""Rank-30 tensor decomposition of 2x3x10 matrix multiplication.

Approach (built from scratch per directive):
  Split the 10-column output into two independent 2x3x5 blocks.
  For each 2x3x5 block, use a Kronecker-product construction that
  factorizes the (m, q) = (3, 5) middle-to-output map as 3 x (1 x 5),
  giving rank m*q = 15 per block.  Total rank = 2 * 15 = 30.

Trace convention: W[k*n + i, r]  (k = output column, i = output row).
"""
import numpy as np

DIMENSIONS = [2, 3, 10]


def _kron(a, b):
    """Explicit Kronecker product of two 2-D integer/half-integer matrices."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ai, aj = a.shape
    bi, bj = b.shape
    out = np.zeros((ai * bi, aj * bj), dtype=float)
    for x in range(ai):
        for y in range(aj):
            out[x * bi:(x + 1) * bi, y * bj:(y + 1) * bj] = a[x, y] * b
    return out


def construct():
    n, m, p = DIMENSIONS            # 2, 3, 10
    q = p // 2                      # 5, columns per block
    nblk = p // q                   # 2 blocks
    r_block = m * q                 # 15, Kronecker rank per 2x3x5 block
    total_rank = nblk * r_block     # 30

    U = np.zeros((n * m, total_rank), dtype=float)   # (6, 30)
    V = np.zeros((m * p, total_rank), dtype=float)   # (30, 30)
    W = np.zeros((p * n, total_rank), dtype=float)   # (20, 30)

    # Kronecker factors for the (m, q) = (3, 5) map.
    #  A is (m, 1): selects the middle index j.
    #  B is (1, q): selects the local output column k_local.
    #  A ⊗ B is (3, 5) with entry (j, k_local) = 1.
    A = np.eye(m)[:, 0:1]          # (3, 1)
    B = np.ones((1, q))            # (1, 5)
    M = _kron(A, B)                # (3, 5), all ones

    for h in range(nblk):
        base = h * r_block
        for j in range(m):
            for k_local in range(q):
                r = base + j * q + k_local
                k_global = h * q + k_local
                # U: index (i, j) -> i*m + j
                for i in range(n):
                    U[i * m + j, r] = M[j, k_local]
                # V: index (j, k_global) -> j*p + k_global
                V[j * p + k_global, r] = M[j, k_local]
                # W: trace convention index (k_global, i) -> k_global*n + i
                for i in range(n):
                    W[k_global * n + i, r] = M[j, k_local]

    return {"U": U, "V": V, "W": W}


# EVOLVE-BLOCK-END
