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
        if time.time() > deadline - 5:  # Stop if running out of time
            break
        
        # Try multiple random initializations for this rank
        max_trials = 5
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small values
            np.random.seed(42 + R * 100 + trial)  # Reproducible
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            # Alternating least squares iterations
            max_iter = 200
            prev_error = np.inf
            for it in range(max_iter):
                if time.time() > deadline - 3:
                    break
                
                # Fix V and W, optimize U
                # For each column r of U, minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||
                # This is equivalent to solving a least squares problem
                # Residual R = T - sum_{r' != r} U_r' ⊗ V_r' ⊗ W_r'
                # Then U_r is the best rank-1 approximation to R in the first mode
                
                # Compute residual for each mode
                # For U optimization:
                # T_reshaped has shape (n*m, m*p*p*n)
                # The contribution of U[:, r] is U[:, r] ⊗ (V[:, r] ⊗ W[:, r])
                # where V[:, r] ⊗ W[:, r] has shape (m*p, p*n)
                
                # More efficient: use the fact that we can compute the Gram matrix
                
                # Let's use a simpler approach: compute the full reconstruction and error
                # Then do a gradient step
                
                # Actually, let's use ALS properly
                # For fixed V and W, U can be optimized by solving:
                # min_U ||T - sum_r U_r ⊗ (V_r ⊗ W_r)||_F^2
                # This is a linear least squares problem for each column of U
                
                # Compute the "flattened" target and basis
                # T_flattened: (n*m, m*p * p*n)
                T_flat = T.reshape(n * m, -1)
                
                # For each r, the basis vector is V[:, r] ⊗ W[:, r] reshaped
                # B_r = (V[:, r] ⊗ W[:, r]).reshape(-1)
                # Then T_flat ≈ sum_r U[:, r] ⊗ B_r
                # This is a linear system: T_flat = U @ B^T where B has rows B_r^T
                
                # Build B matrix: (R, m*p*p*n)
                B = np.zeros((R, m * p * p * n))
                for r in range(R):
                    B[r, :] = (V[:, r] * W[:, r].reshape(1, -1)).reshape(-1)
                    # Wait, this isn't right. V[:, r] has shape (m*p,), W[:, r] has shape (p*n,)
                    # The outer product V[:, r] ⊗ W[:, r] has shape (m*p, p*n)
                    # Reshaped to (m*p*p*n,)
                    B[r, :] = (np.outer(V[:, r], W[:, r])).reshape(-1)
                
                # Now solve: T_flat ≈ U @ B^T
                # This is a linear least squares: min_U ||T_flat - U @ B^T||_F^2
                # Solution: U = T_flat @ (B^T @ B)^{-1} @ B^T  (if B^T @ B is invertible)
                # Or use np.linalg.lstsq
                
                try:
                    U_new, _, _, _ = np.linalg.lstsq(B.T, T_flat, rcond=None)
                    U = U_new
                except:
                    pass
                
                # Similarly for V and W
                # For V: T_flat_v = T.reshape(m*p, n*m*p*n)
                # Basis: for each r, (U[:, r] ⊗ W[:, r]) reshaped
                T_flat_v = T.reshape(m * p, -1)
                B_v = np.zeros((R, n * m * p * n))
                for r in range(R):
                    B_v[r, :] = (np.outer(U[:, r], W[:, r])).reshape(-1)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(B_v.T, T_flat_v, rcond=None)
                    V = V_new
                except:
                    pass
                
                # For W: T_flat_w = T.reshape(p*n, n*m*m*p)
                T_flat_w = T.reshape(p * n, -1)
                B_w = np.zeros((R, n * m * m * p))
                for r in range(R):
                    B_w[r, :] = (np.outer(U[:, r], V[:, r])).reshape(-1)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(B_w.T, T_flat_w, rcond=None)
                    W = W_new
                except:
                    pass
                
                # Compute error
                recon = np.zeros_like(T)
                for r in range(R):
                    recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                error = np.linalg.norm(T - recon)
                
                if abs(prev_error - error) < 1e-10:
                    break
                prev_error = error
            
            # Check if we found an exact (or near-exact) decomposition
            if prev_error < 1e-6:
                # Try to round to rational numbers (integers or half-integers)
                # Check if rounding gives an exact decomposition
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
                    break  # Found a better rank, stop searching for this R
        
        # If we found a better rank, we could break early, but let's continue to see if we can do even better
        # Actually, for time efficiency, if we find one, we can break
        if best_rank < rank_school:
            # Check if we should continue searching for lower ranks
            if best_rank == lower_bound:
                break
            # Continue to try lower ranks
    
    # Return the best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
