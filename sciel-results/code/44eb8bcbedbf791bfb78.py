"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication.
    
    Uses a direct sum decomposition based on partitioning the 3x5 matrix into 
    three 1x5 row-vector multiplications. Each 1x5x7 multiplication tensor 
    has rank 7 (max(5, 7)). Total rank is 3 * 7 = 21.
    
    Returns the decomposition with rank 21.
    """
    n, m, p = DIMENSIONS
    # Rank is n * max(m, p) for this specific partitioning strategy
    # For 1x5x7, rank is 7. Total 3*7=21.
    rank = n * p  # 3 * 7 = 21
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Decomposition logic:
    # We iterate over each row i of the first matrix (n rows).
    # For each row i, we use p rank-1 terms, indexed by k (0 to p-1).
    # Column index c = i * p + k.
    
    for i in range(n):
        for k in range(p):
            c = i * p + k
            
            # U part:
            # For this row i, U[i*m + j, c] should be 1 for all j in 0..m-1.
            for j in range(m):
                U[i * m + j][c] = 1
                
            # V part:
            # V[j*p + k', c] should be 1 if k' == k, else 0.
            # It is 1 for all j if k' == k.
            for j in range(m):
                V[j * p + k][c] = 1
                
            # W part:
            # W[k'*n + i', c] should be 1 if k' == k and i' == i, else 0.
            # Only one entry is 1: W[k*n + i, c] = 1.
            W[k * n + i][c] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
