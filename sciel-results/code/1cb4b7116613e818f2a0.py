import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m + j, j*p + k, k*n + i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Pre-compute flattened views of T for ALS
    T_flat_u = T.reshape(n * m, -1) # (nm, mp*pn)
    T_flat_v = T.reshape(m * p, -1) # (mp, nm*pn)
    T_flat_w = T.reshape(p * n, -1) # (pn, nm*mp)
    
    # Schoolbook decomposition (Rank 48)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Lower bound: max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower bound up to schoolbook - 1
    # Focus heavily on the lower bound as it is the theoretical minimum for 3x4x4
    for R in range(lower_bound, rank_school):
        if time.time() > deadline - 5:
            break
            
        # If we are at the lower bound, use more trials and SVD init
        if R == lower_bound:
            max_trials = 100
        else:
            max_trials = 20
            
        found_better = False
        
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + R * 1000 + trial * 7)
            
            # Initialization Strategy:
            # 1. SVD-based for the first few trials
            # 2. Random for the rest
            
            if trial < 5:
                # SVD Initialization
                # Unfold T to get matrices, take top R singular vectors
                U, _, _ = np.linalg.svd(T_flat_u, full_matrices=False)
                U = U[:, :R]
                
                V, _, _ = np.linalg.svd(T_flat_v, full_matrices=False)
                V = V[:, :R]
                
                W, _, _ = np.linalg.svd(T_flat_w, full_matrices=False)
                W = W[:, :R]
            else:
                # Random Initialization with scaling
                # Scale to match the norm of the tensor
                scale = np.linalg.norm(T) / (n * m * p)
                U = np.random.randn(n * m, R) * scale
                V = np.random.randn(m * p, R) * scale
                W = np.random.randn(p * n, R) * scale

            # Alternating least squares iterations
            max_iter = 500
            prev_error = np.inf
            current_error = np.inf
            
            for it in range(max_iter):
                if time.time() > deadline - 3:
                    break
                
                # Optimize U
                # B_u[r] = (V[:, r] ⊗ W[:, r]).reshape(-1)
                # We can construct B_u more efficiently using broadcasting or loops
                # Since R is small, a loop is fine
                B_u = np.zeros((R, m * p * p * n))
                for r in range(R):
                    B_u[r] = np.outer(V[:, r], W[:, r]).reshape(-1)
                
                try:
                    U, _, _, _ = np.linalg.lstsq(B_u.T, T_flat_u, rcond=None)
                except:
                    pass
                
                # Optimize V
                B_v = np.zeros((R, n * m * p * n))
                for r in range(R):
                    B_v[r] = np.outer(U[:, r], W[:, r]).reshape(-1)
                    
                try:
                    V, _, _, _ = np.linalg.lstsq(B_v.T, T_flat_v, rcond=None)
                except:
                    pass
                
                # Optimize W
                B_w = np.zeros((R, n * m * m * p))
                for r in range(R):
                    B_w[r] = np.outer(U[:, r], V[:, r]).reshape(-1)
                    
                try:
                    W, _, _, _ = np.linalg.lstsq(B_w.T, T_flat_w, rcond=None)
                except:
                    pass
                
                # Compute error
                recon = np.zeros_like(T)
                for r in range(R):
                    recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                current_error = np.linalg.norm(T - recon)
                
                # Check for convergence
                if abs(prev_error - current_error) < 1e-10:
                    break
                prev_error = current_error
            
            # Check if we found an exact (or near-exact) decomposition
            if current_error < 1e-8:
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                recon_rounded = np.zeros_like(T)
                for r in range(R):
                    recon_rounded += np.einsum('a,b,c->abc', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
                error_rounded = np.linalg.norm(T - recon_rounded)
                
                if error_rounded < 1e-6:
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    best_rank = R
                    found_better = True
                    break
        
        # If we found a better rank, stop searching for higher ranks
        if found_better:
            break
    
    # Return the best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
