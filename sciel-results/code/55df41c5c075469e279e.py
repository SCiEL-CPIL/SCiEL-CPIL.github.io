"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

import time
import numpy as np
from itertools import product

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    DEADLINE = 115  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Verify a decomposition
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Block Strassen decomposition:
    # Split A (4x5) into blocks: A11 (2x2), A12 (2x3), A21 (2x2), A22 (2x3)
    # Split B (5x6) into blocks: B11 (2x3), B12 (2x3), B21 (3x3), B22 (3x3)
    # C = A*B has blocks: C11 (2x3), C12 (2x3), C21 (2x3), C22 (2x3)
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # For each 2x2 block multiplication, we can use Strassen if blocks are square
    # But here we have mixed sizes. Let's use a different approach.
    
    # Alternative: Use the fact that we can express each output block as a sum
    # of products of sub-blocks, and then decompose each such product.
    
    # Let's try a direct rank-1 decomposition with some clever grouping.
    # The key is to find a set of rank-1 tensors that span the multiplication tensor.
    
    # For now, return schoolbook as a valid baseline
    return best_result

# EVOLVE-BLOCK-END
