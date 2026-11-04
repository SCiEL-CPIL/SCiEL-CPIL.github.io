import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    rank = n * m * p
    
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A index
                idx_a = i * m + j
                # B index
                idx_b = j * p + k
                # C index (W convention: k*n + i)
                idx_c = k * n + i
                
                U[idx_a, col] = 1.0
                V[idx_b, col] = 1.0
                W[idx_c, col] = 1.0
                col += 1
                
    # Time guard: return best result if approaching limit
    # (Construction is instantaneous, so this is a formality)
    if time.time() - start_time > 115:
        pass

    # Convert to lists as required by interface
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
