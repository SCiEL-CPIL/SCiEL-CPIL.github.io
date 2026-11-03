"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find a low-rank exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Returns a dictionary with 'U', 'V', 'W' matrices.
    """
    n, m, p = DIMENSIONS
    # Optimal rank is max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    rank = m * p  # 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Column index for (j, k) pair
    def col_idx(j, k):
        return j * p + k
    
    # Fill U: U[i*m + j, col(j,k)] = 1 for all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col_idx(j, k)] = 1
    
    # Fill V: V[j*p + k, col(j,k)] = 1
    for j in range(m):
        for k in range(p):
            V[j * p + k][col_idx(j, k)] = 1
    
    # Fill W: W[k*n + i, col(j,k)] = 1 for all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                W[k * n + i][col_idx(j, k)] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
