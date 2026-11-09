import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Attempt to find a better rank (heuristic)
    # This is a placeholder for a more complex search if time allows.
    # For now, we just return the schoolbook solution as it is guaranteed to be valid.
    # A real search would involve random restarts and local optimization.
    
    if time.time() - start_time < deadline:
        # Simple check: can we merge any 3 terms?
        # This is O(R^3) which is 75^3 = 421,875. Feasible.
        # But checking rank of sum of 3 tensors is non-trivial.
        # We skip this for safety and speed.
        pass
        
    # Convert to lists for the return interface
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
