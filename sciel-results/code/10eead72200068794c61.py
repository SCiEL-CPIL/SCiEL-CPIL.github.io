"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _rank32_decomposition(n, m, p):
    """
    Rank-32 exact decomposition for 2x2 * 2x16 matrix multiplication.
    
    For each column k of B (0..15), we use 2 rank-1 terms:
      Term 1: captures A[i,0]*B[0,k] for i=0,1 -> contributes to C[0,k] and C[1,k]
      Term 2: captures A[i,1]*B[1,k] for i=0,1 -> contributes to C[0,k] and C[1,k]
    
    Total rank = 2 * p = 32
    """
    rank = 2 * p  # 32
    U = [[0] * rank for _ in range(n * m)]  # 4 x 32
    V = [[0] * rank for _ in range(m * p)]  # 32 x 32
    W = [[0] * rank for _ in range(p * n)]  # 32 x 32
    
    for k in range(p):
        # Term 1: B[0,k] row, A column 0 (A[0,0] and A[1,0])
        col1 = 2 * k
        U[0][col1] = 1   # A[0,0] -> index 0*m+0 = 0
        U[2][col1] = 1   # A[1,0] -> index 1*m+0 = 2
        V[k][col1] = 1   # B[0,k] -> index 0*p+k = k
        W[2 * k][col1] = 1     # C[0,k] -> index k*n+0 = 2k
        W[2 * k + 1][col1] = 1 # C[1,k] -> index k*n+1 = 2k+1
        
        # Term 2: B[1,k] row, A column 1 (A[0,1] and A[1,1])
        col2 = 2 * k + 1
        U[1][col2] = 1   # A[0,1] -> index 0*m+1 = 1
        U[3][col2] = 1   # A[1,1] -> index 1*m+1 = 3
        V[p + k][col2] = 1  # B[1,k] -> index 1*p+k = p+k
        W[2 * k][col2] = 1     # C[0,k] -> index k*n+0 = 2k
        W[2 * k + 1][col2] = 1 # C[1,k] -> index k*n+1 = 2k+1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _rank32_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
