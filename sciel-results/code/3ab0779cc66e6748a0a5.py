import numpy as np
import time

def construct():
    n, m, p = 3, 3, 7
    rank = n * m * p
    # Use float to be safe
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    start_time = time.time()
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, idx] = 1.0
                V[v_idx, idx] = 1.0
                W[w_idx, idx] = 1.0
                idx += 1
                
                if time.time() - start_time > 110:
                    break
            if time.time() - start_time > 110:
                break
        if time.time() - start_time > 110:
            break
            
    return {"U": U, "V": V, "W": W}
