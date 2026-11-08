import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    
    # Deadline guard (formality, this is fast)
    start_time = time.time()
    deadline = start_time + 145.0  # 150s limit, 5s margin
    
    # Rank-n*m decomposition
    rank = n * m
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r in range(rank):
        if time.time() > deadline:
            break
            
        i = r // m
        j = r % m
        
        # Hypothesis: Consistent Column-Major indexing for all factors
        # U (n x m): index = j * n + i
        u_idx = j * n + i
        U[u_idx, r] = 1.0
        
        # V (m x p): index = k * m + j
        # W (p x n): index = k * n + i (as per prompt)
        for k in range(p):
            v_idx = k * m + j
            V[v_idx, r] = 1.0
            
            w_idx = k * n + i
            W[w_idx, r] = 1.0
            
    return {"U": U, "V": V, "W": W}
