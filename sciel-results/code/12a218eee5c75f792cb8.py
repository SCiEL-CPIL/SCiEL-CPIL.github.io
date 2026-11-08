"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook decomposition
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to list of lists as expected
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Try to see if we can do better with a quick search
    # For now, return schoolbook as it's guaranteed correct
    # and finding a better decomposition is non-trivial
    
    if time.time() > deadline - 5:
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Quick attempt: try rank = rank_schoolbook - 1 by checking if we can
    # find a linear dependency that allows us to remove one term
    # This is a heuristic and may not succeed
    
    # For 4x4 x 4x6, the schoolbook rank is 96.
    # Without a known better decomposition, return schoolbook.
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
