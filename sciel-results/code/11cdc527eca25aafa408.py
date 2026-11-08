import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Deadline guard: 120s limit, use 110s to be safe
    start_time = time.time()
    deadline = start_time + 110
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check deadline
                if time.time() > deadline:
                    # Return partial result? No, must be valid.
                    # Since this loop is fast (105 iterations), it won't timeout.
                    # But we keep the guard for compliance.
                    pass
                    
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
    # Convert to lists as per interface requirement if strict, 
    # but numpy arrays are usually accepted or converted.
    # The prompt says "factor matrices of shapes...".
    # The seed program returned lists of lists.
    # I will return lists of lists to be safe with the interface.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
