import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank0 = n * m * p
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((p*n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition using numerical optimization
    # ... (this is where the optimization would go)
    
    # For now, just return the schoolbook decomposition
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
