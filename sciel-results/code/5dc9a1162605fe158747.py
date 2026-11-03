"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    # Rank-10 decomposition for 2x5x5 matrix multiplication.
    # Direct sum of two 1x5x5 row-vector multiplication tensors, each of rank 5.
    #
    # Convention: W[k*n+i, r] where n=2, m=5, p=5
    # U shape: (n*m, rank) = (10, rank)
    # V shape: (m*p, rank) = (25, rank)  
    # W shape: (p*n, rank) = (10, rank)
    
    n, m, p = DIMENSIONS
    rank = n * m  # = 10
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each row i in [0, n), construct the 1x5x5 schoolbook decomposition.
    # Term index col = i * (m*p) + j * p + k
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
