"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import random

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank: n*m*p = 60
    schoolbook_rank = n * m * p
    
    # Try to find a lower-rank decomposition using a greedy/rank-revealing approach
    # For now, return the schoolbook decomposition as a safe baseline
    # The tensor entries are T[i,j,k] = 1 if we're computing (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Convention: W[k*n+i, r]
    
    rank = schoolbook_rank
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
