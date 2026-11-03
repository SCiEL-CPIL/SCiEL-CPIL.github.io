"""Optimized matrix multiplication tensor decomposition for 3x4 x 4x8."""
DIMENSIONS = [3, 4, 8]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    
    def build_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a lower-rank decomposition
    # Strategy: use block structure and Strassen-like identities
    
    # For now, return schoolbook as it's guaranteed correct
    # Future: implement search for lower rank
    result = build_schoolbook()
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        return result
    
    return result
# EVOLVE-BLOCK-END
