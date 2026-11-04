"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # 110s safety margin within 120s limit
    n, m, p = DIMENSIONS  # n=3, m=5, p=5
    
    # Build the target tensor T[i,j,k] = 1 for all valid indices
    # The matrix multiplication tensor has T[i,j,k] = 1 if the (i,j,k) term exists
    target = np.ones((n, m, p))
    
    # Start with schoolbook rank-75 decomposition
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-8):
        """Verify that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k"""
        rank = U.shape[1]
        reconstructed = np.zeros((n, m, p))
        for c in range(rank):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            # Add outer product contribution
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        reconstructed[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        target = np.ones((n, m, p))
        return np.max(np.abs(reconstructed - target)) < tol
    
    # Try to find lower rank decompositions using numerical optimization
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    # Attempt rank reduction using iterative optimization
    # Try ranks from 70 down to 65 (known that rank 69 exists for 3x5x5)
    for target_rank in [70, 69, 68, 67, 66, 65]:
        if time.time() > deadline - 30:  # Keep 30s buffer
            break
        
        # Initialize with random perturbation of schoolbook solution
        # Use a subset of the schoolbook terms
        U_init = U_school[:, :target_rank].copy()
        V_init = V_school[:, :target_rank].copy()
        W_init = W_school[:, :target_rank].copy()
        
        # Add small random noise to escape local minima
        np.random.seed(42)
        U_init += np.random.normal(0, 0.1, U_init.shape)
        V_init += np.random.normal(0, 0.1, V_init.shape)
        W_init += np.random.normal(0, 0.1, W_init.shape)
        
        # Define objective: minimize ||sum_c U_c ⊗ V_c ⊗ W_c - T||_F^2
        def objective(params):
            if time.time() > deadline - 10:
                return 1e10
            U_flat, V_flat, W_flat = params[:n*m*target_rank], params[n*m*target_rank:2*n*m*target_rank+m*p*target_rank], params[2*n*m*target_rank+m*p*target_rank:]
            U = U_flat.reshape(n * m, target_rank)
            V = V_flat.reshape(m * p, target_rank)
            W = W_flat.reshape(p * n, target_rank)
            
            reconstructed = np.zeros((n, m, p))
            for c in range(target_rank):
                u = U[:, c].reshape(n, m)
                v = V[:, c].reshape(m, p)
                w = W[:, c].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            reconstructed[i, j, k] += u[i, j] * v[j, k] * w[k, i]
            
            diff = reconstructed - target
            return np.sum(diff ** 2)
        
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(42 + attempt)
            U_init = U_school[:, :target_rank].copy()
            V_init = V_school[:, :target_rank].copy()
            W_init = W_school[:, :target_rank].copy()
            
            # Scale down to allow room for optimization
            U_init *= 0.9
            V_init *= 0.9
            W_init *= 0.9
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = minimize(objective, params_init, method='L-BFGS-B', 
                                 options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
                
                if result.fun < 1e-6:
                    U_opt = result.x[:n*m*target_rank].reshape(n * m, target_rank)
                    V_opt = result.x[n*m*target_rank:2*n*m*target_rank+m*p*target_rank].reshape(m * p, target_rank)
                    W_opt = result.x[2*n*m*target_rank+m*p*target_rank:].reshape(p * n, target_rank)
                    
                    if verify_decomposition(U_opt, V_opt, W_opt, n, m, p, tol=1e-6):
                        best_rank = target_rank
                        best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                        break  # Found valid decomposition, try lower rank
            except Exception:
                continue
    
    return best_result
# EVOLVE-BLOCK-END
