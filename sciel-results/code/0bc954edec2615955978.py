import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Vectorized construction of schoolbook decomposition
    ii, jj, kk = np.indices((n, m, p))
    flat = np.arange(rank)
    
    u_idx = ii.ravel() * m + jj.ravel()
    v_idx = jj.ravel() * p + kk.ravel()
    w_idx = kk.ravel() * n + ii.ravel()
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    U[u_idx, flat] = 1
    V[v_idx, flat] = 1
    W[w_idx, flat] = 1
    
    if time.time() > deadline:
        # Should not happen, but safety
        pass
    
    return {"U": U, "V": V, "W": W}
