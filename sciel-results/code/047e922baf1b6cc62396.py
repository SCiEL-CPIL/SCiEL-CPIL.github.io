import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize arrays with zeros
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, column] = 1.0
                V[v_idx, column] = 1.0
                W[w_idx, column] = 1.0
                column += 1
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # If we broke early due to deadline, the decomposition is incomplete/invalid.
    # However, schoolbook is O(nmp) which is tiny (84 iterations).
    # So this guard is just for compliance with the directive.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
