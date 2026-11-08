"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit to be safe
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # If we hit the deadline, we might have an incomplete decomposition.
    # But for n=3, m=3, p=7, the loop is very fast (63 iterations).
    # So this guard is just for safety.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
