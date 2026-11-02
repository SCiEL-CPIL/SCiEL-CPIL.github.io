import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    rank = n * m * p
    
    # Create index arrays for all (i, j, k) triples
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    # Build factor matrices
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    U[i_idx * m + j_idx, np.arange(rank)] = 1.0
    V[j_idx * p + k_idx, np.arange(rank)] = 1.0
    W[k_idx * n + i_idx, np.arange(rank)] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
