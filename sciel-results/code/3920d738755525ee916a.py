"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Trivial rank-n*m*p decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Check if we have time to try a better decomposition
    if time.time() < deadline - 20:
        # Try to find a lower-rank decomposition using structured search
        # For now, the trivial solution is returned
        pass
    
    return result
# EVOLVE-BLOCK-END
