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
    for R in range(lower_bound, rank_school):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations for this rank
        # Increase trials if we have time
        max_trials = 10
        found_better = False
        
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            np.random.seed(42 + R * 100 + trial)
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            # Alternating least squares iterations
            max_iter = 300
            prev_error = np.inf
            for it in range(max_iter):
                if time.time() > deadline - 3:
                    break
                
                # Optimize U
                # B_u[r] = (V[:, r] ⊗ W[:, r]).reshape(-1)
                B_u = np.array([np.outer(V[:, r], W[:, r]).reshape(-1) for r in range(R)])
                try:
                    U, _, _, _ = np.linalg.lstsq(B_u.T, T_flat_u, rcond=None)
                except:
                    pass
                
                # Optimize V
                B_v = np.array([np.outer(U[:, r], W[:, r]).reshape(-1) for r in range(R)])
                try:
                    V, _, _, _ = np.linalg.lstsq(B_v.T, T_flat_v, rcond=None)
                except:
                    pass
                
                # Optimize W
                B_w = np.array([np.outer(U[:, r], V[:, r]).reshape(-1) for r in range(R)])
                try:
                    W, _, _, _ = np.linalg.lstsq(B_w.T, T_flat_w, rcond=None)
                except:
                    pass
                
                # Compute error
                recon = np.zeros_like(T)
                for r in range(R):
                    recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                error = np.linalg.norm(T - recon)
                
                if abs(prev_error - error) < 1e-12:
                    break
                prev_error = error
            
            # Check if we found an exact (or near-exact) decomposition
            if prev_error < 1e-6:
                # Try to round to half-integers
                # Try rounding to nearest 0.5
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
