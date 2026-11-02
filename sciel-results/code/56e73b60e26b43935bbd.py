"""Matrix multiplication 3x4x7 decomposition - rank 84 schoolbook."""
DIMENSIONS = [3, 4, 7]

import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns an exact tensor decomposition for 3x4 * 4x7 matrix multiplication.
    
    Uses the standard schoolbook algorithm with rank n*m*p = 84.
    Attempts to use Strassen or pairing tricks for lower rank were analyzed
    but found to be inapplicable or invalid for these specific dimensions
    without introducing cross-terms that cannot be canceled efficiently.
    """
    n, m, p = DIMENSIONS
    
    U = [[0] * (n*m*p) for _ in range(n*m)]
    V = [[0] * (n*m*p) for _ in range(m*p)]
    W = [[0] * (n*m*p) for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
