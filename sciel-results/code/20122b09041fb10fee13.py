import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    naive_rank = n * m * p
    
    # Start with the naive decomposition
    U_naive = np.zeros((n * m, naive_rank))
    V_naive = np.zeros((m * p, naive_rank))
    W_naive = np.zeros((p * n, naive_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i * m + j, col] = 1
                V_naive[j * p + k, col] = 1
                W_naive[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
    best_rank = naive_rank
    
    # Try to find lower rank decompositions using continuous optimization
    # with tensor train decomposition approach
    
    # We'll try various bond dimensions and optimize
    target_ranks = [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
        
        # Initialize with random small perturbations around the naive solution
        # truncated to target_rank columns
        U_init = np.random.randn(n * m, target_rank) * 0.1
        V_init = np.random.randn(m * p, target_rank) * 0.1
        W_init = np.random.randn(p * n, target_rank) * 0.1
        
        # Add some structure from the naive solution
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = min(i * m * p + j * p + k, target_rank - 1)
                    U_init[i * m + j, idx] += 1.0
                    V_init[j * p + k, idx] += 1.0
                    W_init[k * n + i, idx] += 1.0
        
        def objective(params):
            U_flat, V_flat, W_flat = np.split(params, [n * m * target_rank, 
                                                        (n * m + m * p) * target_rank])
            U_opt = U_flat.reshape(n * m, target_rank)
            V_opt = V_flat.reshape(m * p, target_rank)
            W_opt = W_flat.reshape(p * n, target_rank)
            
            # Compute the tensor: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # We want this to equal 1 if (i,j,k) corresponds to a valid multiplication term
            # and 0 otherwise
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # This term should be 1 in the multiplication tensor
                        val = np.sum(U_opt[i * m + j, :] * V_opt[j * p + k, :] * W_opt[k * n + i, :])
                        error += (val - 1.0) ** 2
            
            # Regularization to encourage sparsity and half-integer values
            reg = 0.01 * (np.sum(U_opt ** 2) + np.sum(V_opt ** 2) + np.sum(W_opt ** 2))
            
            return error + reg
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = minimize(objective, params_init, method='L-BFGS-B', 
                            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.success and result.fun < 1e-6:
                U_opt, V_opt, W_opt = np.split(result.x, [n * m * target_rank, 
                                                          (n * m + m * p) * target_rank])
                U_opt = U_opt.reshape(n * m, target_rank)
                V_opt = V_opt.reshape(m * p, target_rank)
                W_opt = W_opt.reshape(p * n, target_rank)
                
                # Round to nearest half-integers
                U_rounded = np.round(2 * U_opt) / 2.0
                V_rounded = np.round(2 * V_opt) / 2.0
                W_rounded = np.round(2 * W_opt) / 2.0
                
                # Verify exact decomposition
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_rounded[i * m + j, :] * V_rounded[j * p + k, :] * 
                                        W_rounded[k * n + i, :])
                            if abs(val - 1.0) > 1e-10:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_rank = target_rank
                    best_result = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    if target_rank == 7:  # Theoretical minimum
                        break
        except Exception:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
