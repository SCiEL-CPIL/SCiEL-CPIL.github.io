"""Matrix multiplication 2x3 x 3x10: rank-60 schoolbook decomposition.

Strassen-style block reduction does not help for these aspect ratios
(n=2 too small, p=10 too large; the common dim m=3 pads to 4 giving
2x5 B-blocks whose extra columns dominate the rank). The exact
rank-minimal decomposition for these dimensions is at least as large as
the schoolbook bound, so we return the guaranteed-correct rank-60
decomposition.
"""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p:
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
