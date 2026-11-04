"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Target tensor: T[(i,j),(j',k),(k',i')] = δ_{j,j'}δ_{k,k'}δ_{i,i'}
    # We need: Σ_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = T[(i,j),(j',k),(k',i')]
    
    # Non-zero: T[(i,j),(j,k),(k,i)] = 1 for i in [0,3), j in [0,4), k in [0,5)
    # That's 3*4*5 = 60 non-zero entries
    
    # Let's use a greedy algorithm with exact rational arithmetic
    # Represent vectors as lists of Fractions
    
    # Initialize residual to the target tensor
    # residual[(i,j),(j',k),(k',i')] = T[...] - sum of added terms
    
    # For efficiency, we'll work with the 60 non-zero constraints and 
    # track the zero constraints separately
    
    # Actually, let's use a different approach: 
    # Use the fact that the tensor has a special structure
    
    # The tensor can be written as a sum of 60 rank-1 tensors (schoolbook).
    # We want to find a shorter sum.
    
    # Let's try a randomized approach: initialize U, V, W randomly with 
    # rational entries, then use alternating least squares to minimize
    # the residual, checking for exactness periodically.
    
    # But this is hard to make exact. Let me try a different strategy.
    
    # Strategy: Use a known upper bound and verify it.
    # For 3x4x5, the rank is at most 45 (by various constructions).
    # Let me try to construct a rank-45 decomposition.
    
    # Actually, let me try a simple improvement: 
    # Group the middle dimension. Since m=4, we can write 4 = 2+2.
    # This might allow a decomposition with rank = 3*(2+2)*5/2 or similar.
    
    # For now, let's return the schoolbook decomposition as a safe baseline.
    # The rank is 60.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
