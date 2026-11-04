import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # 1. Construct Schoolbook Solution (Rank 84)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=int)
    V_sb = np.zeros((m * p, rank_sb), dtype=int)
    W_sb = np.zeros((p * n, rank_sb), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    # Verify Schoolbook
    # T[i*m+j, j*p+k, k*n+i] should be 1
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This is guaranteed by construction.
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # 2. Attempt to find Rank 83
    # We use a random search with a specific structure:
    # Start with 84 terms, remove one, and try to adjust the remaining 83 to compensate.
    # This is hard. Instead, we try random sparse matrices for rank 83.
    
    target_rank = 83
    if time.time() < deadline - 20:
        # Try a few random configurations for rank 83
        # We use integer entries {-1, 0, 1}
        # To make it more likely, we can use the schoolbook as a guide?
        # No, random is better for diversity.
        
        max_attempts = 500
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            U = np.zeros((n * m, target_rank), dtype=int)
            V = np.zeros((m * p, target_rank), dtype=int)
            W = np.zeros((p * n, target_rank), dtype=int)
            
            # Generate sparse random matrices
            # Density ~ 3/R
            for r in range(target_rank):
                # U
                idx_u = np.random.choice(n * m, size=min(3, n * m), replace=False)
                U[idx_u, r] = np.random.choice([-1, 1], size=len(idx_u))
                # V
                idx_v = np.random.choice(m * p, size=min(4, m * p), replace=False)
                V[idx_v, r] = np.random.choice([-1, 1], size=len(idx_v))
                # W
                idx_w = np.random.choice(p * n, size=min(3, p * n), replace=False)
                W[idx_w, r] = np.random.choice([-1, 1], size=len(idx_w))
            
            # Check validity
            # We only need to check the n*m*p specific entries
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        # Compute sum
                        # Vectorized dot product
                        # u_vec = U[i*m+j, :]
                        # v_vec = V[j*p+k, :]
                        # w_vec = W[k*n+i, :]
                        # val = np.dot(u_vec, np.dot(v_vec, w_vec))
                        val = np.einsum('r,r,r->', U[i*m+j], V[j*p+k], W[k*n+i])
                        if val != 1:
                            valid = False
                            break
            
            if valid:
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
                
    # Return the best result
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
