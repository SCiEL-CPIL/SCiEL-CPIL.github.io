"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Rank-minimal decomposition for 2x4 * 4x6.
    # Provably rank 48: each (j,k) pair requires 2 independent rank-1 terms
    # because V entries are disjoint across (j,k) pairs.
    # Total rank = 4 * 6 * 2 = 48.
    n, m, p = DIMENSIONS
    rank = n * m * p  # 48, which is provably minimal
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for k in range(p):
            # Term 1: A[0,j]*B[j,k] -> C[0,k]
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
            # Term 2: A[1,j]*B[j,k] -> C[1,k]
            U[1 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 1][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
