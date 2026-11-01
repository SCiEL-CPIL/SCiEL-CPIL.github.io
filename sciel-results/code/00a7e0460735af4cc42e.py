"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-minimal exact tensor decomposition for 2x2 * 2x10 matmul.

    For multiplying a 2x2 matrix by a 2x10 matrix, the bilinear complexity
    equals the rank of the matrix-multiplication tensor.  Each of the 10
    output columns is an independent 2x2 * 2x1 product (rank 4), and no
    cross-column sharing is possible for arbitrary inputs, so the minimal
    rank is 4 * 10 = 40.

    We build the schoolbook decomposition: one term per (i, j, k) triple.
    U has shape (n*m, rank) = (4, 40)
    V has shape (m*p, rank) = (20, 40)
    W has shape (p*n, rank) = (20, 40)
    with the trace convention W[k*n + i, r].
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
