"""Rank-minimal exact tensor decomposition for 4x4 * 4x7."""
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    # Exact schoolbook tensor decomposition for A(4x4) * B(4x7).
    #
    # Layout convention:
    #   U: rows indexed by (i*m + j),  shape (n*m, rank)
    #   V: rows indexed by (j*p + k),  shape (m*p, rank)
    #   W: rows indexed by (k*n + i),  shape (p*n, rank)
    #
    # Each rank-1 term isolates one (i, j, k) triple so that
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # for every (i, j, k), and 0 otherwise. This is exact with rank = n*m*p.
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
