import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 150.0  # seconds
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U_sb[i * m + j, c] = 1
                V_sb[j * p + k, c] = 1
                W_sb[k * n + i, c] = 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find a lower rank
    # Lower bound estimate
    min_rank_to_try = 50
    
    for r in range(min_rank_to_try, rank_schoolbook):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random starts for this rank
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random integers in {-1, 0, 1} or similar
            U = np.random.randint(-1, 2, size=(n * m, r))
            V = np.random.randint(-1, 2, size=(m * p, r))
            W = np.random.randint(-1, 2, size=(p * n, r))
            
            # Check if it's a valid solution
            # We need sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1 for all i,j,k
            # Vectorize:
            # U shape (nm, r), V shape (mp, r), W shape (pn, r)
            # We want T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
            
            # Reshape U to (n, m, r)
            U_reshaped = U.reshape(n, m, r)
            V_reshaped = V.reshape(m, p, r)
            W_reshaped = W.reshape(p, n, r)
            
            # Compute T
            # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
            # This is a batched trace-like operation.
            # U[i,j,c] * V[j,k,c] * W[k,i,c]
            # Sum over c.
            
            # We can do this by iterating over i, j, k or using einsum
            # T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
            
            T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
            
            if np.all(T == 1):
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break # Found minimal rank for this r, try lower r? No, r is increasing.
                # Actually, we want minimal r. So we should try r from low to high.
                # If we find one, we can break the outer loop if we are confident it's minimal?
                # No, we should continue to see if we can find lower?
                # But r is increasing. So the first one found is the lowest for this range.
                # But we started at 50. Maybe there is a solution at 40?
                # We should search from low to high.
        
        # If we found a solution at r, we can break?
        # No, we want the absolute minimum.
        # But searching from 1 to 90 is too slow.
        # We assume the minimum is close to the schoolbook or known bounds.
        # Let's just return the best found.
        
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
