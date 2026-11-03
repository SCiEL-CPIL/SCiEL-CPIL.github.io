"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Leave 20s safety margin for 120s limit
    
    # Start with schoolbook rank
    best_rank = n * m * p
    best_U = np.zeros((n * m, best_rank))
    best_V = np.zeros((m * p, best_rank))
    best_W = np.zeros((p * n, best_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j, col] = 1
                best_V[j * p + k, col] = 1
                best_W[k * n + i, col] = 1
                col += 1
    
    # Try to reduce rank using structured ansatz based on block decomposition
    # For 4x5 * 5x5, we can try to exploit the rectangular structure
    # The minimum rank is at least max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    # But typically it's higher. Let's try a few structured approaches.
    
    # Approach 1: Try to use the fact that n=4, p=5 to find cancellations
    # We'll use a simple heuristic: try random low-rank approximations and verify
    
    # For now, keep the schoolbook solution as it's guaranteed correct
    # A proper Strassen-like approach for these dimensions would require
    # careful algebraic construction which is complex for non-square cases
    
    # Convert to lists for the required output format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
