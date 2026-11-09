"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import itertools
import numpy as np
from functools import lru_cache

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    # Rank: n * p * (m+1) = 3 * 8 * 4 = 96
    rank = n * p * (m + 1)  # 96
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Index into U, V, W columns
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # This gives us n*m*p = 72 entries with all 1s
                # But we need a structured decomposition that might have rank <= 72
                # Actually, let's use the standard approach but try to find a lower rank
                pass
    
    # Let's use the tensor product approach:
    # The multiplication M_{3,3,8} can be seen as M_{3,3,1} ⊗ M_{1,3,8} (not quite)
    # Or more precisely: M_{3,3,8} has a block structure
    # A (3x3) * B (3x8) = C (3x8)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The schoolbook rank is 3*3*8 = 72.
    # Can we do better?
    # For M_{3,3,3}, the optimal rank is 23 (not 27).
    # So M_{3,3,8} should be <= 8 * 23 / 3 * something... 
    # Actually, M_{3,3,8} can be decomposed using 8 copies of M_{3,3,1} with a shared middle dimension
    # Or better: think of it as 8 independent M_{3,3,1} multiplications but sharing the A matrix structure
    # This doesn't directly help.
    
    # Let's try a Winograd-style approach:
    # For each output C[i,k], C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    # Using Winograd for size 3: 
    #   a = A[i,0] + A[i,1], b = A[i,1] + A[i,2]
    #   m1 = a * B[j=0,k]... no this is for convolution.
    
    # Let's just use the schoolbook decomposition with rank 72 for now,
    # and structure it cleanly. The key is to get a VALID decomposition first.
    
    rank = n * m * p  # 72
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
