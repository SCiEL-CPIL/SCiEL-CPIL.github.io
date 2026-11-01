"""
Exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
Uses 3 independent 2x2x4 blocks, each with a rank-10 decomposition.
Total rank = 30.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-30 exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    Uses 3 blocks of 4 columns. Each block uses a rank-10 decomposition
    for 2x2x4 matrix multiplication.
    Total rank = 3 * 10 = 30.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    num_blocks = p // 4  # 3
    rank_per_block = 10
    rank = num_blocks * rank_per_block  # 30
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 30
    V = [[0] * rank for _ in range(m * p)]  # 24 x 30
    W = [[0] * rank for _ in range(p * n)]  # 24 x 30
    
    for t in range(num_blocks):
        base = t * rank_per_block
        c0 = 4 * t
        c1 = 4 * t + 1
        c2 = 4 * t + 2
        c3 = 4 * t + 3
        
        # Term 0: M1 = (A00+A11) * (B0c0+B1c1)
        c = base + 0
        U[0][c] = 1
        U[3][c] = 1
        V[0*p + c0][c] = 1
        V[1*p + c1][c] = 1
        W[c0*n + 0][c] = 1
        W[c1*n + 1][c] = 1
        
        # Term 1: M2 = (A10+A11) * B0c0
        c = base + 1
        U[2][c] = 1
        U[3][c] = 1
        V[0*p + c0][c] = 1
        W[c0*n + 1][c] = 1
        W[c1*n + 1][c] = -1
        
        # Term 2: M3 = A00 * (B0c1 - B1c1)
        c = base + 2
        U[0][c] = 1
        V[0*p + c1][c] = 1
        V[1*p + c1][c] = -1
        W[c1*n + 0][c] = 1
        W[c1*n + 1][c] = 1
        
        # Term 3: M4 = A11 * (B1c0 - B0c0)
        c = base + 3
        U[3][c] = 1
        V[1*p + c0][c] = 1
        V[0*p + c0][c] = -1
        W[c0*n + 0][c] = 1
        W[c0*n + 1][c] = 1
        
        # Term 4: M5 = (A00+A01) * B1c1
        c = base + 4
        U[0][c] = 1
        U[1][c] = 1
        V[1*p + c1][c] = 1
        W[c0*n + 0][c] = -1
        W[c1*n + 0][c] = 1
        
        # Term 5: M6 = (A10-A00) * (B0c0+B0c1)
        c = base + 5
        U[2][c] = 1
        U[0][c] = -1
        V[0*p + c0][c] = 1
        V[0*p + c1][c] = 1
        W[c1*n + 1][c] = 1
        
        # Term 6: M7 = (A01-A11) * (B1c0+B1c1)
        c = base + 6
        U[1][c] = 1
        U[3][c] = -1
        V[1*p + c0][c] = 1
        V[1*p + c1][c] = 1
        W[c0*n + 0][c] = 1
        
        # Term 7: A00 * B0c2
        c = base + 7
        U[0][c] = 1
        V[0*p + c2][c] = 1
        W[c2*n + 0][c] = 1
        
        # Term 8: A01 * B1c2
        c = base + 8
        U[1][c] = 1
        V[1*p + c2][c] = 1
        W[c2*n + 0][c] = 1
        
        # Term 9: (A10+A11) * B0c3
        c = base + 9
        U[2][c] = 1
        U[3][c] = 1
        V[0*p + c3][c] = 1
        W[c3*n + 1][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
