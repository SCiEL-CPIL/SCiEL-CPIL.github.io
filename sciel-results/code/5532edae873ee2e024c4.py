"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Use a rank-48 decomposition by combining terms
    # For each (i,k), use 2 terms instead of 4
    r = 48
    
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Build the decomposition: for each i in range(n), k in range(p),
    # we have 2 terms corresponding to r = 2*(i*p + k) and r = 2*(i*p + k) + 1
    for i in range(n):
        for k in range(p):
            base_r = 2 * (i * p + k)
            # Term 1: covers j=0,1
            # A_{i,0}B_{0,k} + A_{i,1}B_{1,k} = (A_{i,0}+A_{i,1})(B_{0,k}+B_{1,k}) - A_{i,0}B_{1,k} - A_{i,1}B_{0,k}
            # This doesn't work directly. Let me use a different approach.
            # Use 4 terms per (i,k) but share across i or k.
            pass
    
    # Fallback: use trivial decomposition with rank 72
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    idx = 0
    for k in range(p):
        for j in range(m):
            for i in range(n):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, idx] = 1
                V[v_idx, idx] = 1
                W[w_idx, idx] = 1
                idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
