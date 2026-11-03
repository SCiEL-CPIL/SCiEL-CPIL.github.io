import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the tensor decomposition for matrix multiplication C = A * B
    where A is 4x4 and B is 4x9.
    Uses the standard schoolbook algorithm, which has a rank of n*m*p.
    For 4x4x9, the rank is 144.
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
