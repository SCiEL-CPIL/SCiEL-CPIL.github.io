"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
DIMENSIONS = [3, 5, 7]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    U0, V0, W0 = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U0, V0, W0
    
    # Vectorized verification
    def verify_fast(U, V, W, tol=1e-6):
        if not (np.all(np.isfinite(U)) and np.all(np.isfinite(V)) and np.all(np.isfinite(W))):
            return False
        R = U.shape[1]
        if R == 0:
            return False
        
        # Create index arrays for all (i,j,k) combinations
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        u_idx = i_idx * m + j_idx  # shape (n*m*p,)
        v_idx = j_idx * p + k_idx  # shape (n*m*p,)
        w_idx = k_idx * n + i_idx  # shape (n*m*p,)
        
        # Get the relevant rows
        u_vals = U[u_idx, :]  # shape (n*m*p, R)
        v_vals = V[v_idx, :]  # shape (n*m*p, R)
        w_vals = W[w_idx, :]  # shape (n*m*p, R)
        
        # Compute sum over r
        prod = u_vals * v_vals * w_vals  # shape (n*m*p, R)
        result = prod.sum(axis=1)  # shape (n*m*p,)
        
        return np.all(np.abs(result - 1.0) < tol)
    
    # Verify schoolbook
    if not verify_fast(U0, V0, W0):
        # Should not happen, but return anyway
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    
    # Try to reduce rank by removing redundant terms
    search_start = time.time()
    max_search_time = 30  # seconds
    
    for trial in range(200):
        if time.time() - search_start > max_search_time:
            break
        if time.time() > deadline - 20:
            break
        
        # Try removing a random term
        if best_rank <= 35:  # Theoretical lower bound max(3*5, 5*7, 3*7) = 35
            break
        
        remove_col = random.randint(0, best_rank - 1)
        U_try = np.delete(best_U, remove_col, axis=1)
        V_try = np.delete(best_V, remove_col, axis=1)
        W_try = np.delete(best_W, remove_col, axis=1)
        
        if verify_fast(U_try, V_try, W_try):
            best_rank -= 1
            best_U, best_V, best_W = U_try, V_try, W_try
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
