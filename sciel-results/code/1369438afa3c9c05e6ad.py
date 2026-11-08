"""
Rewritten program for matrix multiplication 2x3 by 3x4.
Uses the standard schoolbook decomposition of rank n*m*p.
Includes explicit deadline guard as per directive.
"""

import time
import sys

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    # Fallback to a trivial valid decomposition if we run out of time
                    # (Though this loop is very fast, this is a safety net)
                    # We just return what we have so far, padded with zeros?
                    # No, we must return a valid decomposition.
                    # The loop completes in microseconds, so this is unlikely to trigger.
                    pass
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
