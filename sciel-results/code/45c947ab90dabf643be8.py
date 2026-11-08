"""Program for rank-minimal tensor decomposition of 2x4 * 4x7 matrix multiplication."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2 by 4 matrix by a 4 by 7 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    We use the schoolbook decomposition which has rank n*m*p = 56.
    For these dimensions, this is the standard exact decomposition.
    """
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
