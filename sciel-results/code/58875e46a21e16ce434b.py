import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Vectorized initialization for speed, though rank remains trivial
    indices = np.arange(rank)
    i_idx = (indices // (m * p)) % n
    j_idx = (indices // p) % m
    k_idx = indices % p
    
    U[i_idx * m + j_idx, indices] = 1
    V[j_idx * p + k_idx, indices] = 1
    W[k_idx * n + i_idx, indices] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
