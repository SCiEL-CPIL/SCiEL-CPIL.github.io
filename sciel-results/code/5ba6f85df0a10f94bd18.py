"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = 2, 5, 6
    
    # Block decomposition:
    # A is 2x5. Split into A1 (2x2, cols 0-1) and A2 (2x3, cols 2-4)
    # B is 5x6. Split into B1 (2x6, rows 0-1) and B2 (3x6, rows 2-4)
    # C = A1*B1 + A2*B2
    
    # For each block multiplication, we use schoolbook.
    # A1*B1: rank = 2*2*6 = 24
    # A2*B2: rank = 2*3*6 = 36
    # Total rank = 24 + 36 = 60 (same as schoolbook, not an improvement)
    
    # Let's try a different split. What if we use Strassen-like for 2x2 blocks?
    # But the dimensions are not powers of 2 in a clean way.
    
    # Actually, let's just do schoolbook but verify it works.
    # The schoolbook rank is n*m*p = 60.
    
    # Is there any known lower bound? For 2x5*5x6, the rank is at least 2*5+2*6+5*6-2 = 10+12+30-2 = 50?
    # Actually the lower bound is n*m + n*p + m*p - n - m - p + 1? No, that's not right.
    
    # Let me just return the schoolbook decomposition to be safe.
    
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
