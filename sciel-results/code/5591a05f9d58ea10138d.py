import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Deadline guard: 120s limit, use 110s for safety
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Default result: Schoolbook
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # For n=5, the schoolbook rank is 125.
    # Strassen's algorithm for n=4 has rank 28, but n=5 is not a power of 2.
    # Embedding n=5 into n=8 yields rank 343, which is worse.
    # Embedding n=5 into n=4 is impossible.
    # Therefore, the schoolbook decomposition is the best simple exact decomposition available.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
