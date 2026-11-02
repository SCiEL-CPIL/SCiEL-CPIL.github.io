"""Rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-56 exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Strategy: Partition into 8 independent 2x2 * 2x2 block multiplications,
    each decomposed via Strassen (rank 7). Total rank = 8 * 7 = 56.
    
    The Winograd convolutional approach was investigated but the 4-element dot
    product in the middle dimension (m=4) cannot be reduced below 4 scalar
    multiplications per output entry in a way that yields a valid exact
    decomposition for the full tensor. The Strassen block approach remains
    the best known exact decomposition at rank 56.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    total_rank = 56
    
    # U: (n*m) x rank = 8 x 56
    # V: (m*p) x rank = 32 x 56  
    # W: (p*n) x rank = 16 x 56
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Index conventions:
    # A[i,j] -> U index: i*m + j
    # B[j,k] -> V index: j*p + k
    # C[i,k] -> W index: k*n + i
    
    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)
            
            # A block indices (2x2 block at column offset 2u)
            a11 = 0 * 4 + 2 * u      
            a12 = 0 * 4 + 2 * u + 1  
            a21 = 1 * 4 + 2 * u      
            a22 = 1 * 4 + 2 * u + 1  
            
            # B block indices (2x2 block at row offset 2u, col offset 2v)
            b11 = 2 * u * 8 + 2 * v      
            b12 = 2 * u * 8 + 2 * v + 1  
            b21 = (2 * u + 1) * 8 + 2 * v      
            b22 = (2 * u + 1) * 8 + 2 * v + 1  
            
            # C block indices (2x2 block at col offset 2v)
            c11 = 4 * v      
            c12 = 4 * v + 2  
            c21 = 4 * v + 1  
            c22 = 4 * v + 3  
            
            # Strassen 7 products for X*Y where X=A[u], Y=B[u,v]
            
            # M1 = (X11+X22)*(Y11+Y22)
            U[a11][base_col] += 1
            U[a22][base_col] += 1
            V[b11][base_col] += 1
            V[b22][base_col] += 1
            W[c11][base_col] += 1
            W[c22][base_col] += 1
            
            # M2 = (X21+X22)*Y11
            U[a21][base_col + 1] += 1
            U[a22][base_col + 1] += 1
            V[b11][base_col + 1] += 1
            W[c21][base_col + 1] += 1
            W[c22][base_col + 1] -= 1
            
            # M3 = X11*(Y12-Y22)
            U[a11][base_col + 2] += 1
            V[b12][base_col + 2] += 1
            V[b22][base_col + 2] -= 1
            W[c12][base_col + 2] += 1
            W[c22][base_col + 2] += 1
            
            # M4 = X22*(Y21-Y11)
            U[a22][base_col + 3] += 1
            V[b21][base_col + 3] += 1
            V[b11][base_col + 3] -= 1
            W[c11][base_col + 3] += 1
            W[c21][base_col + 3] += 1
            
            # M5 = (X11+X12)*Y22
            U[a11][base_col + 4] += 1
            U[a12][base_col + 4] += 1
            V[b22][base_col + 4] += 1
            W[c11][base_col + 4] -= 1
            W[c12][base_col + 4] += 1
            
            # M6 = (X21-X11)*(Y11+Y12)
            U[a21][base_col + 5] += 1
            U[a11][base_col + 5] -= 1
            V[b11][base_col + 5] += 1
            V[b12][base_col + 5] += 1
            W[c22][base_col + 5] += 1
            
            # M7 = (X12-X22)*(Y21+Y22)
            U[a12][base_col + 6] += 1
            U[a22][base_col + 6] -= 1
            V[b21][base_col + 6] += 1
            V[b22][base_col + 6] += 1
            W[c11][base_col + 6] += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
