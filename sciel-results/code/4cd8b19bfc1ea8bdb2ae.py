"""Matrix multiplication 3x4x8 - rank-minimal tensor decomposition search."""

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 96
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1.0
                V_schoolbook[j * p + k, col] = 1.0
                W_schoolbook[k * n + i, col] = 1.0
                col += 1
    
    best_U = U_schoolbook.copy()
    best_V = V_schoolbook.copy()
    best_W = W_schoolbook.copy()
    best_rank = rank_schoolbook
    
    # The tensor T has entries T[i, j, k] = 1 for all valid (i,j,k)
    # We want: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Define the residual function
    def make_residual(n, m, p, rank):
        """Create a residual function for given dimensions and rank."""
        # Precompute index mappings
        U_idx = np.array([i * m + j for i in range(n) for j in range(m) for k in range(p)])
        V_idx = np.array([j * p + k for i in range(n) for j in range(m) for k in range(p)])
        W_idx = np.array([k * n + i for i in range(n) for j in range(m) for k in range(p)])
        
        def residual(params):
            # params is flattened [U, V, W]
            U = params[:n*m*rank].reshape(n*m, rank)
            V = params[n*m*rank:2*n*m*rank].reshape(m*p, rank)
            W = params[2*n*m*rank:].reshape(p*n, rank)
            
            # Compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # This is element-wise product summed over c
            terms = U[U_idx] * V[V_idx] * W[W_idx]
            result = np.sum(terms, axis=1)  # sum over rank dimension
            return result - 1.0  # residual from target value 1
        
        return residual
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # Known results suggest rank might be around 80-90 for this case
    
    np.random.seed(42)
    
    # Strategy: Try ranks from 80 down to schoolbook-1
    # Use random initialization and local optimization
    
    for target_rank in range(80, min(rank_schoolbook, 90)):
        if time.time() > deadline - 20:
            break
        
        # Create residual function
        residual_fn = make_residual(n, m, p, target_rank)
        
        # Try multiple random initializations
        best_local_rank = None
        best_local_params = None
        best_local_residual = np.inf
        
        num_trials = 5
        for trial in range(num_trials):
            if time.time() > deadline - 15:
                break
            
            # Random initialization
            U_init = np.random.randn(n * m, target_rank) * 0.1
            V_init = np.random.randn(m * p, target_rank) * 0.1
            W_init = np.random.randn(p * n, target_rank) * 0.1
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Compute initial residual
            res = residual_fn(params_init)
            initial_residual = np.sum(res**2)
            
            # Use L-BFGS-B for optimization
            try:
                result = minimize(
                    lambda x: np.sum(residual_fn(x)**2),
                    params_init,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                final_residual = np.sum(residual_fn(result.x)**2)
                
                if final_residual < best_local_residual and final_residual < 1e-6:
                    best_local_residual = final_residual
                    best_local_params = result.x
                    best_local_rank = target_rank
            except Exception:
                pass
        
        if best_local_rank is not None and best_local_residual < 1e-6:
            # Extract U, V, W from params
            U = best_local_params[:n*m*target_rank].reshape(n*m, target_rank)
            V = best_local_params[n*m*target_rank:2*n*m*target_rank].reshape(m*p, target_rank)
            W = best_local_params[2*n*m*target_rank:].reshape(p*n, target_rank)
            
            # Verify the decomposition
            U_idx = np.array([i * m + j for i in range(n) for j in range(m) for k in range(p)])
            V_idx = np.array([j * p + k for i in range(n) for j in range(m) for k in range(p)])
            W_idx = np.array([k * n + i for i in range(n) for j in range(m) for k in range(p)])
            
            terms = U[U_idx] * V[V_idx] * W[W_idx]
            result_check = np.sum(terms, axis=1)
            
            if np.allclose(result_check, 1.0, atol=1e-5):
                best_U = U
                best_V = V
                best_W = W
                best_rank = target_rank
                break  # Found a valid lower rank decomposition
    
    # If no better rank found, try a structured approach
    # For 3x4x8, we can potentially exploit the structure
    # Let's try a Winograd-like approach for the 4x8 part
    
    if best_rank == rank_schoolbook:
        # Try a block-based decomposition
        # Split the 4x8 matrix into 4x4 blocks
        # Use Strassen-like methods for 4x4 if possible
        
        # For now, keep the schoolbook if we can't find better
        pass
    
    # Convert to lists for the required format
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

