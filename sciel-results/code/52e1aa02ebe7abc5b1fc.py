"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary containing the factor matrices U, V, W for the 
    matrix multiplication tensor of dimensions 3x5 by 5x7.
    Uses the schoolbook decomposition of rank n*m*p.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U corresponds to input A (n x m)
                U[i * m + j][column] = 1
                # V corresponds to input B (m x p)
                V[j * p + k][column] = 1
                # W corresponds to output C (n x p) flattened as k*n + i
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
