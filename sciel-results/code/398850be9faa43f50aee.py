"""
Program to find a rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication.
Returns the schoolbook decomposition of rank 125, which is the standard trivial upper bound.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s deadline as per contract

    n, m, p = DIMENSIONS
    rank = n * m * p  # 125 for 5x5x5

    # We return the schoolbook decomposition which has rank 125.
    # U shape: (n*m, rank) = (25, 125)
    # V shape: (m*p, rank) = (25, 125)
    # W shape: (p*n, rank) = (25, 125)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Index mapping for schoolbook:
    # Each term in the sum corresponds to a unique (i, j, k) triple.
    # r corresponds to (i,j,k) -> col = i * m * p + j * p + k
    # u_r has 1 at index (i*m + j)
    # v_r has 1 at index (j*p + k)
    # w_r has 1 at index (k*n + i)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                row_u = i * m + j
                row_v = j * p + k
                row_w = k * n + i
                
                U[row_u, col] = 1.0
                V[row_v, col] = 1.0
                W[row_w, col] = 1.0

    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
