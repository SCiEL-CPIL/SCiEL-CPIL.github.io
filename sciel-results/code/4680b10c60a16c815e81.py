"""Matrix multiplication 3x5x5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """Schoolbook decomposition: rank = n*m*p = 75.

    For each triple (i, j, k) with i in [0,n), j in [0,m), k in [0,p),
    create one rank-1 term with:
        U[i*m + j]  = 1 at that term's column
        V[j*p + k]  = 1 at that term's column
        W[k*n + i]  = 1 at that term's column
    This exactly reproduces C[i,k] = sum_j A[i,j] * B[j,k].
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
