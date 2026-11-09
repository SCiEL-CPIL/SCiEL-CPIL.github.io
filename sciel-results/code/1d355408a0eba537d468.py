import numpy as np

def construct():
    n, m, p = 3, 3, 6
    rank = 18
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Decompose 3x3 * 3x6 into three 3x3 * 3x2 blocks
    # Each 3x3 * 3x2 block has rank 6
    # Total rank 18
    
    for q in range(3):
        for i in range(n):
            for j_local in range(2):
                t = q * 6 + i * 2 + j_local
                j_global = q * 2 + j_local
                
                # U[i, k] = delta(i, k)
                U[i * m + i, t] = 1.0
                
                # V[k, j] = delta(k, i) * delta(j, j_global)
                V[i * p + j_global, t] = 1.0
                
                # W[j, i] = 1 (for the specific i, j_global)
                # Contracted W index is j_global * n + i
                W[j_global * n + i, t] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
