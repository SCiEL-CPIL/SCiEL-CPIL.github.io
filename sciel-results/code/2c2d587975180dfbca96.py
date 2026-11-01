DIMENSIONS = [3, 3, 3]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Exact decomposition for 3x3 matrix multiplication.
    Returns factor matrices U, V, W with the trace convention W[k*n+i, r].
    """
    n, m, p = 3, 3, 3
    
    # Start with the trivial rank-27 decomposition
    rank = 27
    U = np.zeros((9, rank), dtype=float)
    V = np.zeros((9, rank), dtype=float)
    W = np.zeros((9, rank), dtype=float)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1.0
                V[j*3+k, col] = 1.0
                W[k*3+i, col] = 1.0
                col += 1
    
    # Verify correctness
    valid = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0.0
                for r in range(rank):
                    s += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
                if abs(s - 1.0) > 1e-12:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback (should never reach here)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
