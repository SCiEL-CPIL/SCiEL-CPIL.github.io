import numpy as np
import time

DIMENSIONS = [3, 5, 7]
TIME_LIMIT = 120.0  # Seconds

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = n * m * p  # 105
    num_u = n * m  # 15
    num_v = m * p  # 35
    num_w = p * n  # 21
    
    U = np.zeros((num_u, rank), dtype=float)
    V = np.zeros((num_v, rank), dtype=float)
    W = np.zeros((num_w, rank), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check time limit
                if time.time() - start_time > TIME_LIMIT - 5.0:
                    # Return current best (which is the trivial solution if we haven't finished, 
                    # but here the loop is short so this won't trigger)
                    return {"U": U, "V": V, "W": W}
                    
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
                r += 1
                
    return {"U": U, "V": V, "W": W}
