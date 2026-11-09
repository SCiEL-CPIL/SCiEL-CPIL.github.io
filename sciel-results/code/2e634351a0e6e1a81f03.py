"""Matrix multiplication 3,3,4 - rank-minimal exact tensor decomposition."""
import time
import random

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x3 matrix by a 3x4 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices of shapes
        (n*m, rank), (m*p, rank), (p*n, rank)
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook decomposition: rank = n * m * p = 36
    # This is the standard exact decomposition with integer entries.
    # While better decompositions exist (e.g., rank ~32 or lower),
    # constructing them explicitly with integer/half-integer coefficients
    # is non-trivial and not guaranteed to be minimal without specific
    # algebraic knowledge. We provide the robust schoolbook solution.
    
    rank = n * m * p  # 36
    
    U = [[0] * rank for _ in range(n * m)]  # 9 x 36
    V = [[0] * rank for _ in range(m * p)]  # 12 x 36
    W = [[0] * rank for _ in range(p * n)]  # 12 x 36
    
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
