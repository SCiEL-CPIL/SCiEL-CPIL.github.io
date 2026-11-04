"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # 140s safety margin within 150s limit
    n, m, p = DIMENSIONS  # n=3, m=5, p=5
    
    # Build the target tensor T[i,j,k] = 1 for all valid indices
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
            reconstructed += u[:, :, None] * v[None, :, :] * w[None, None, :]
        return np.max(np.abs(reconstructed - target)) < tol
    
    # Try to find lower rank decompositions using numerical optimization
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    # Attempt rank reduction using iterative optimization
    # Try ranks from 70 down to 65 (known that rank 69 exists for 3x5x5)
    for target_rank in [70, 69, 68, 67, 66, 65]:
        if time.time() > deadline - 20:  # Keep 20s buffer
            break
        
        # Define objective: minimize ||sum_c U_c ⊗ V_c ⊗ W_c - T||_F^2
        def objective(params):
            if time.time() > deadline - 5:
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
                reconstructed += u[:, :, None] * v[None, :, :] * w[None, None, :]
            
            diff = reconstructed - target
            return np.sum(diff ** 2)
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + attempt * 17)
            
            # Random initialization with structure
            U_init = np.random.normal(0, 1, (n * m, target_rank))
            V_init = np.random.normal(0, 1, (m * p, target_rank))
            W_init = np.random.normal(0, 1, (p * n, target_rank))
            
            # Also try perturbed schoolbook
            if attempt % 2 == 0:
                U_init = U_school[:, :target_rank].copy()
                V_init = V_school[:, :target_rank].copy()
                W_init = W_school[:, :target_rank].copy()
                U_init += np.random.normal(0, 0.5, U_init.shape)
                V_init += np.random.normal(0, 0.5, V_init.shape)
                W_init += np.random.normal(0, 0.5, W_init.shape)
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = minimize(objective, params_init, method='L-BFGS-B', 
                                 options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 50})
                
                if result.fun < 1e-8:
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
