import numpy as np

def construct():
    n, m, p = 3, 3, 6
    rank = 18
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Block 1: k in 0..2
    for r in range(9):
        i = r // 3
        j = r % 3
        U[i * m + j, r] = 1
        for k in range(3):
            V[j * p + k, r] = 1
            W[k * n + i, r] = 1
            
    # Block 2: k in 3..5
    for r in range(9):
        r_idx = r + 9
        i = r // 3
        j = r % 3
        U[i * m + j, r_idx] = 1
        for k in range(3, 6):
            V[j * p + k, r_idx] = 1
            W[k * n + i, r_idx] = 1
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
