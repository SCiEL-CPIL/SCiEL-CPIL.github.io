"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
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
    
    # Try ranks from lower_bound+1 up to rank_school-1
    # For each rank, try a few random initializations with ALS
    for R in range(lower_bound + 1, rank_school):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations for this rank
        max_trials = 3
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small values
            np.random.seed(42 + R * 1000 + trial * 100)
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            # Alternating least squares iterations
            max_iter = 50
            prev_error = np.inf
            converged = False
            
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
                
                # More efficient: use the fact that
                # sum_r U[:, r] * (V[:, r] * W[:, r]) = T
                # For fixed V, W, the optimal U is:
                # U = T_mat @ (V * W)^+ where T_mat is T reshaped to (n*m, m*p*p*n)
                # and (V * W) is the element-wise product reshaped to (R, m*p*p*n)
                
                # Reshape T to (n*m, m*p*p*n)
                T_mat = T.reshape(n * m, m * p * p * n)
                
                # Compute V * W for each r: (R, m*p, p*n) -> (R, m*p*p*n)
                VW = (V * W[:, np.newaxis, :]).reshape(R, m * p * p * n)
                
                # Solve for U: U = T_mat @ VW^+
                # Use least squares
                try:
                    U_new, _, _, _ = np.linalg.lstsq(VW.T, T_mat.T, rcond=None)
                    U_new = U_new.T
                except:
                    break
                
                # Fix U and W, optimize V
                UW = (U * W[np.newaxis, :, :]).reshape(R, n * m * p * n)
                T_mat2 = T.reshape(m * p, n * m * p * n)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(UW.T, T_mat2.T, rcond=None)
                    V_new = V_new.T
                except:
                    break
                
                # Fix U and V, optimize W
                UV = (U * V[np.newaxis, :, :]).reshape(R, n * m * m * p)
                T_mat3 = T.reshape(p * n, n * m * m * p)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(UV.T, T_mat3.T, rcond=None)
                    W_new = W_new.T
                except:
                    break
                
                # Check convergence
                U = U_new
                V = V_new
                W = W_new
                
                # Compute error
                error = 0.0
                for r in range(R):
                    contrib = np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
                    error += np.sum((T - contrib) ** 2)
                
                if abs(prev_error - error) < 1e-12:
                    converged = True
                    break
                prev_error = error
            
            # Check if we found an exact decomposition
            if converged or it >= max_iter - 1:
                # Verify exactness
                max_err = 0.0
                for r in range(R):
                    contrib = np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
                    max_err = max(max_err, np.max(np.abs(T - contrib)))
                
                # Actually, we need to check that the SUM equals T
                # Let's recompute the sum
                T_recon = np.zeros_like(T)
                for r in range(R):
                    T_recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
                
                max_err = np.max(np.abs(T - T_recon))
                
                if max_err < 1e-10:  # Exact enough
                    # Round to nearest half-integer
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify rounded version
                    T_recon2 = np.zeros_like(T)
                    for r in range(R):
                        T_recon2 += np.outer(np.outer(U_rounded[:, r], V_rounded[:, r]), W_rounded[:, r])
                    
                    if np.max(np.abs(T - T_recon2)) < 1e-10:
                        best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                        best_rank = R
                        break  # Found a valid decomposition for this rank
        
        if best_rank < rank_school:
            break  # Found a better rank, stop searching
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
