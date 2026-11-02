"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank n*m decomposition using the "outer product" structure
    # C[i,k] = sum_j A[i,j] B[j,k]
    # For each j, we can separate the i dependence.
    # Rank = n * m
    rank = n * m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for i in range(n):
            # U[i*m + j, col] = 1
            U[i * m + j][col] = 1
            
            # V[j*p + k, col] = 1 for all k
            for k in range(p):
                V[j * p + k][col] = 1
                
            # W[k*n + i, col] = 1 for all k
            for k in range(p):
                W[k * n + i][col] = 1
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
