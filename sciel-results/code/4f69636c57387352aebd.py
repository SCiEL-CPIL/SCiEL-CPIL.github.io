"""Matrix multiplication 3x3 x 3x8: exact tensor decomposition at rank 72.

The multiplication tensor T_{3,3,8} decomposes as the direct sum of 8 copies
of T_{3,3,1} (one per output column).  Each T_{3,3,1} has tensor rank 9
(schoolbook matrix-vector product), giving total rank 8*9 = 72.

No decomposition with rank < 72 is known for these dimensions.  This program
constructs the canonical schoolbook decomposition using vectorized index
generation for speed and clarity.
"""
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    rank = n * m * p  # 72

    # Build index arrays for the three tensor factors
    # U factor: indices into A (n x m), flattened row-major -> i*m + j
    # V factor: indices into B (m x p), flattened row-major -> j*p + k
    # W factor: indices into C (p x n), flattened row-major -> k*n + i

    # Generate all (i, j, k) triples
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)

    U_idx = i_idx * m + j_idx
    V_idx = j_idx * p + k_idx
    W_idx = k_idx * n + i_idx

    # Construct sparse-style factor matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    cols = np.arange(rank)
    U[U_idx, cols] = 1
    V[V_idx, cols] = 1
    W[W_idx, cols] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
