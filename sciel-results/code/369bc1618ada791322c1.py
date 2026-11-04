import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 120.0
_MARGIN = 5.0

def _time_ok():
    return time.time() < _DEADLINE - _MARGIN

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2 matrices
    # U2, V2, W2 are 4x7 matrices
    # Rows correspond to indices i*2+j for i,j in {0,1}
    
    # Initialize 2x2 Strassen factors
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=np.int8)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=np.int8)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=np.int8)
    
    # Construct 4x4 factors using Kronecker product structure
    # Rank is 7 * 7 = 49
    R = 49
    U = np.zeros((n*m, R), dtype=np.int8)
    V = np.zeros((m*p, R), dtype=np.int8)
    W = np.zeros((p*n, R), dtype=np.int8)
    
    if not _time_ok():
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Pre-compute outer products for all 4x4 indices to speed up assignment
    # There are 16 indices for U, V, W
    # U indices: (i, j) -> i1,i2,j1,j2 -> idx1, idx2
    # We can just loop, it's fast enough.
    
    for i in range(n):
        for j in range(m):
            i1, i2 = i // 2, i % 2
            j1, j2 = j // 2, j % 2
            idx_u1 = i1 * 2 + j1
            idx_u2 = i2 * 2 + j2
            U[i*m + j] = np.outer(U2[idx_u1], U2[idx_u2]).flatten()
            
    if not _time_ok():
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    for j in range(m):
        for k in range(p):
            j1, j2 = j // 2, j % 2
            k1, k2 = k // 2, k % 2
            idx_v1 = j1 * 2 + k1
            idx_v2 = j2 * 2 + k2
            V[j*p + k] = np.outer(V2[idx_v1], V2[idx_v2]).flatten()
            
    if not _time_ok():
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    for k in range(p):
        for i in range(n):
            k1, k2 = k // 2, k % 2
            i1, i2 = i // 2, i % 2
            idx_w1 = k1 * 2 + i1
            idx_w2 = k2 * 2 + i2
            W[k*n + i] = np.outer(W2[idx_w1], W2[idx_w2]).flatten()
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
