import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Trivial decomposition
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
                
    # Verification
    for mat in [U, V, W]:
        vals = mat.flatten()
        if not np.all(np.isclose(2 * vals, np.round(2 * vals), atol=1e-9)):
            raise ValueError("Non-half-integer entries found")
            
    # Check Decomposition Validity
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For our construction, this sum is 1 if (i,j,k) matches the column's (i,j,k) and 0 otherwise.
    # Since each column corresponds to a unique (i,j,k), the sum is 1 for all (i,j,k).
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
