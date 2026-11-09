"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 54 for schoolbook
    
    # Build the schoolbook decomposition
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
    
    # Quick verification (optional, can be skipped for speed)
    # Verify that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k
    # For schoolbook, this is trivially true by construction
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
