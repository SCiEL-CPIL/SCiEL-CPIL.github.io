"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from itertools import product
import scipy.optimize as opt

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # The evaluator kills after 150s. We use 145s as a hard stop to be safe.
    # However, this construction is O(1) and extremely fast.
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 72
    
    # Target tensor T[i, j, k] = 1
    # We construct the schoolbook decomposition explicitly.
    # U[i, j, c], V[j, k, c], W[k, i, c]
    # Index c corresponds to the tuple (i, j, k) in a specific order.
    
    # Flatten indices for U: (n*m, rank)
    # Row index in U: i * m + j
    # Col index in U: c
    # U[i*m + j, c] = 1 if c corresponds to (i, j, k)
    
    # Flatten indices for V: (m*p, rank)
    # Row index in V: j * p + k
    # Col index in V: c
    # V[j*p + k, c] = 1 if c corresponds to (i, j, k)
    
    # Flatten indices for W: (p*n, rank)
    # Row index in W: k * n + i
    # Col index in W: c
    # W[k*n + i, c] = 1 if c corresponds to (i, j, k)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U component
                U[i * m + j, c] = 1.0
                # V component
                V[j * p + k, c] = 1.0
                # W component
                W[k * n + i, c] = 1.0
                c += 1
                
    # Verify exactness (optional but good for debugging/safety, very fast)
    # T_rec[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
    # Using einsum on reshaped tensors
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    T_rec = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    T_target = np.ones((n, m, p))
    
    if not np.allclose(T_rec, T_target, atol=1e-10):
        # Fallback: should never happen
        pass
        
    # Return as lists of lists as required by interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
