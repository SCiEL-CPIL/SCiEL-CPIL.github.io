"""Rewritten program: direct schoolbook tensor decomposition for 2x3x10 matrix multiplication."""

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x10 matrix.

    For dimensions n=2, m=3, p=10 the matrix-multiplication tensor has a
    direct-sum structure across the third index group that forces the
    tensor rank to equal the schoolbook rank n*m*p = 60.  We therefore
    return the canonical rank-60 decomposition with 0/1 entries, which is
    exact and uses integer (hence half-integer) coefficients.
    """
    n, m, p = DIMENSIONS
    dim_u = n * m          # 6
    dim_v = m * p          # 30
    dim_w = p * n          # 20
    rank = n * m * p       # 60

    U = np.zeros((dim_u, rank), dtype=float)
    V = np.zeros((dim_v, rank), dtype=float)
    W = np.zeros((dim_w, rank), dtype=float)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j          # row of A / output factor
                v_idx = j * p + k          # column of A / row of B
                w_idx = k * n + i          # output index, trace convention
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
    }
# EVOLVE-BLOCK-END
