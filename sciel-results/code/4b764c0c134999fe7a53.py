import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    # Return whatever is constructed so far, or the full schoolbook if it's fast enough
                    # Since the loop is tiny, this is unlikely to trigger, but good practice.
                    break
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
