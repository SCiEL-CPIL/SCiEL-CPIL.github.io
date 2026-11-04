"""Matrix multiplication 3,5,7 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a tensor decomposition for 3x5 * 5x7 matrix multiplication.
    Uses schoolbook decomposition with rank = 3*5*7 = 105.
    """
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Schoolbook decomposition: rank = n*m*p = 105
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
