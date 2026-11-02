import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Theoretical rank analysis:
    # The tensor T_{2x3, 3x7, 2x7} decomposes into 7 independent copies of T_{2x3, 3x1, 2x1}.
    # Each T_{2x3, 3x1, 2x1} decomposes into 2 independent copies of T_{1x3, 3x1, 1x1} (dot product).
    # The rank of the dot product of length 3 is 3.
    # Total rank = 7 * 2 * 3 = 42.
    rank = n * m * p
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Convert to lists as required by the interface (lists of lists)
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
