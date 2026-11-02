"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Strategy: Use block decomposition to reduce rank below schoolbook (125).
    # Split 5x5 into 3x3 and 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where A11 is 3x3, A12 is 3x2, A21 is 2x3, A22 is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Use Strassen for 3x3 blocks (7 mults) and direct for others.
    # But we need an explicit tensor decomposition.
    
    # Let's build the decomposition by combining:
    # 1. Strassen decomposition for the 3x3 part
    # 2. Direct multiplication for the remaining blocks
    
    # Strassen's 3x3 decomposition:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For the full 5x5, we have:
    # C11 (3x3) = A11*B11 + A12*B21
    # C12 (3x2) = A11*B12 + A12*B22
    # C21 (2x3) = A21*B11 + A22*B21
    # C22 (2x2) = A21*B12 + A22*B22
    
    # Let's enumerate all the terms:
    # For C11 = A11*B11 + A12*B21:
    #   A11*B11: 3x3x3 = 27 terms (but we use Strassen, so 7)
    #   A12*B21: 3x2x3 = 18 terms
    # For C12 = A11*B12 + A12*B22:
    #   A11*B12: 3x3x2 = 18 terms
    #   A12*B22: 3x2x2 = 12 terms
    # For C21 = A21*B11 + A22*B21:
    #   A21*B11: 2x3x3 = 18 terms
    #   A22*B21: 2x2x3 = 12 terms
    # For C22 = A21*B12 + A22*B22:
    #   A21*B12: 2x3x2 = 12 terms
    #   A22*B22: 2x2x2 = 4 terms
    
    # Total with Strassen for A11*B11: 7 + 18 + 18 + 12 + 18 + 12 + 12 + 4 = 101
    # This is better than 125 but still not optimal.
    
    # Let's try to use Strassen more aggressively.
    # Actually, let's just implement the schoolbook decomposition first,
    # then try to optimize.
    
    # For now, return the schoolbook decomposition as it's guaranteed correct.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
