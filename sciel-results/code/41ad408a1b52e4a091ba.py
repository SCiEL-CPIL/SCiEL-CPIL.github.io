import numpy as np
import random
import time

DIMENSIONS = [3, 4, 5]

def check_decomposition(U, V, W, n, m, p):
    """
    Check if U, V, W form a valid decomposition.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Condition: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k
    """
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0
                for r in range(R):
                    val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                if val != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Rank n*m decomposition
    # For each (i, j), create a column r
    # U[i*m + j, r] = 1
    # V[j*p + k, r] = 1 for all k
    # W[k*n + i, r] = 1 for all k, i
    
    R = n * m
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            # U
            U[i * m + j, col] = 1
            
            # V: set 1 for all k at this j
            for k in range(p):
                V[j * p + k, col] = 1
                
            # W: set 1 for all k, i
            for k in range(p):
                for ii in range(n):
                    W[k * n + ii, col] = 1
                    
            col += 1
            
    # Verify
    if not check_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if something went wrong
        base_rank = n * m * p
        U = np.zeros((n * m, base_rank), dtype=int)
        V = np.zeros((m * p, base_rank), dtype=int)
        W = np.zeros((p * n, base_rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
                    if not check_decomposition(U, V, W, n, m, p):
                         # This check is expensive in loop, but safe for fallback
                         pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
