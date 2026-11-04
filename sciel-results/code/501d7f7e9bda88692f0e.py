import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = m * p
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for r in range(rank):
        j = r // p
        k = r % p
        
        # V is identity-like for the (j,k) index
        V[j * p + k, r] = 1
        
        # U has 1s for all i at column j
        for i in range(n):
            U[i * m + j, r] = 1
            
        # W has 1s for all i at row k
        for i in range(n):
            W[k * n + i, r] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
