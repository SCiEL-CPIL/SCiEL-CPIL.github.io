import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def check_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast check using vectorized operations."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                val = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[k, i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def try_remove_term(U, V, W, n, m, p, r_to_remove, tol=1e-9):
    """Try removing term r_to_remove and check if decomposition still holds."""
    rank = U.shape[1]
    mask = np.ones(rank, dtype=bool)
    mask[r_to_remove] = False
    
    U_new = U[:, mask]
    V_new = V[:, mask]
    W_new = W[:, mask]
    
    return check_decomposition_fast(U_new, V_new, W_new, n, m, p, tol)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Initial rank-45 decomposition
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
    
    current_rank = rank
    
    # Greedy removal with multiple passes
    max_passes = 50
    for _pass in range(max_passes):
        if time.time() > deadline:
            break
        
        removed_any = False
        for r in range(current_rank):
            if time.time() > deadline:
                break
            
            if try_remove_term(U, V, W, n, m, p, r):
                # Remove term r
                mask = np.ones(current_rank, dtype=bool)
                mask[r] = False
                U = U[:, mask]
                V = V[:, mask]
                W = W[:, mask]
                current_rank -= 1
                removed_any = True
                break  # Restart the loop since indices changed
        
        if not removed_any:
            break
    
    # If we still have time, try more aggressive search
    if time.time() < deadline - 30 and current_rank > 30:
        # Try swapping terms to enable further removal
        # This is a heuristic: try replacing a term with a linear combination
        pass  # Skip for now to keep it simple
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
