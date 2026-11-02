import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # New approach: Construct using explicit loops to ensure clarity and potential for future optimization hooks
    # Although the rank is the same, this structure allows for easier modification to non-trivial decompositions
    # if coefficients were available.
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                
                U[u_row, idx] = 1
                V[v_row, idx] = 1
                W[w_row, idx] = 1
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
