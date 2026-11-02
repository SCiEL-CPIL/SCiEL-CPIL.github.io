"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    The output factor follows the official trace convention W[k*n+i, r].
    Returns dict with keys 'U', 'V', 'W' as lists of lists with integer entries.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # Schoolbook decomposition: rank = n*m*p = 36
    # U has shape (n*m, rank) = (9, 36)
    # V has shape (m*p, rank) = (12, 36)  
    # W has shape (p*n, rank) = (12, 36)
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
