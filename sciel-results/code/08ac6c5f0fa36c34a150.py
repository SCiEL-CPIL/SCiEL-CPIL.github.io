"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a low-rank decomposition using continuous optimization
    # Start with a rank slightly above the lower bound
    lower_bound = max(n*m, m*p, n*p)
    
    # We will try ranks from lower_bound up to n*m*p
    # But we will use a smarter search: continuous optimization
    
    best_rank = n * m * p
    best_decomp = None
    
    # Try a few ranks
    for rank in range(lower_bound, best_rank + 1):
        # Limit the number of trials to keep time low
        found = _find_decomp_opt(n, m, p, rank)
        if found is not None:
            best_rank = rank
            best_decomp = found
            break
            
    if best_decomp is None:
        return _schoolbook(n, m, p)
        
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _find_decomp_opt(n, m, p, rank):
    """
    Try to find a rank decomposition using continuous optimization.
    We minimize the squared error of the tensor decomposition.
    """
    # Target tensor
    T_target = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Flatten the target tensor for easier handling? 
    # Or use einsum in the error function.
    
    best_error = float('inf')
    best_params = None
    
    # Number of trials
    num_trials = 50
    
    for trial in range(num_trials):
        # Initialize parameters: U, V, W flattened
        # U: (n*m, rank), V: (m*p, rank), W: (n*p, rank)
        # Total params: rank * (n*m + m*p + n*p)
        
        U_init = np.random.randn(n*m, rank) * 0.5
        V_init = np.random.randn(m*p, rank) * 0.5
        W_init = np.random.randn(n*p, rank) * 0.5
        
        # Flatten into a single vector for optimization
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def error_func(x):
            U = x[:n*m*rank].reshape(n*m, rank)
            V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W = x[n*m*rank + m*p*rank:].reshape(n*p, rank)
            
            # Compute T_computed
            # T_computed[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            T_comp = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(T_comp - T_target)
            return error**2
        
        # Use Nelder-Mead or L-BFGS-B
        # L-BFGS-B is better for large numbers of parameters
        # But Nelder-Mead doesn't require gradients.
        # Given the non-convexity, Nelder-Mead might get stuck.
        # Let's try a few steps of gradient descent manually or use scipy minimize.
        
        try:
            res = minimize(error_func, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
            final_error = res.fun
            if final_error < best_error:
                best_error = final_error
                best_params = res.x
                
                # If error is very small, try to round to half-integers
                if final_error < 1e-6:
                    U_opt = best_params[:n*m*rank].reshape(n*m, rank)
                    V_opt = best_params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_opt = best_params[n*m*rank + m*p*rank:].reshape(n*p, rank)
                    
                    # Try rounding to nearest half-integer
                    U_half = np.round(U_opt * 2) / 2
                    V_half = np.round(V_opt * 2) / 2
                    W_half = np.round(W_opt * 2) / 2
                    
                    # Check exactness
                    T_check = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
                    if np.allclose(T_check, T_target, atol=1e-9):
                        return (U_half, V_half, W_half)
                        
        except Exception as e:
            pass
            
    return None
# EVOLVE-BLOCK-END
