import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook solution
    # Rank is n * m * p
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Verify exactness (optional but good for safety)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # recon = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For the schoolbook, each term c corresponds to a unique (i,j,k).
    # U[i*m+j, c] is 1 only if c matches (i,j,k).
    # So the sum is 1 for the matching triple, 0 otherwise.
    # This is exact.
    
    # Convert to lists of floats
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
