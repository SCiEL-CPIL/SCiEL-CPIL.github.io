# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 3, 3
    rank = n * m * p
    
    # Vectorized construction of the schoolbook decomposition indices
    # This approach uses NumPy's array operations to generate the indices
    # for the non-zero entries in U, V, and W, avoiding explicit Python loops.
    
    # Generate all indices from 0 to rank-1
    idx = np.arange(rank)
    
    # Decompose linear index into (i, j, k) coordinates
    # Order in loop: i (outer), j (middle), k (inner)
    # idx = i * (m * p) + j * p + k
    k = idx % p
    tmp = idx // p
    j = tmp % m
    i = tmp // m
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Set the 1.0 entries using advanced indexing
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    U[i * m + j, idx] = 1.0
    V[j * p + k, idx] = 1.0
    W[k * n + i, idx] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
