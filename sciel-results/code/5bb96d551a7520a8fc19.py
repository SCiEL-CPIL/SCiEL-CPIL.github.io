"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # leave margin before 150s wall-clock limit
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    schoolbook_rank = n * m * p  # 60
    
    # Schoolbook decomposition as fallback
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Build the full tensor as a 3D array for verification
    def build_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1
        return T
    
    T = build_tensor()
    
    def verify_decomposition(U_mat, V_mat, W_mat):
        """Check if U * V * W (tensor product) equals T exactly"""
        recon = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        diff = np.abs(recon - T)
        max_diff = np.max(diff) if recon.size > 0 else 0
        return max_diff < 1e-8, max_diff

    def objective(params, R):
        U_size = n*m*R
        V_size = m*p*R
        U_flat = params[:U_size].reshape(n*m, R)
        V_flat = params[U_size:U_size+V_size].reshape(m*p, R)
        W_flat = params[U_size+V_size:].reshape(p*n, R)
        recon = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
        error = np.sum((recon - T)**2)
        return error
    
    def objective_grad(params, R):
        U_size = n*m*R
        V_size = m*p*R
        U_mat = params[:U_size].reshape(n*m, R)
        V_mat = params[U_size:U_size+V_size].reshape(m*p, R)
        W_mat = params[U_size+V_size:].reshape(p*n, R)
        
        recon = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        error = np.sum((recon - T)**2)
        diff = recon - T
        
        grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V_mat, W_mat)
        grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U_mat, W_mat)
        grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U_mat, V_mat)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return error, grad

    # Try to find solutions for ranks below schoolbook using gradient-based optimization
    # with multiple restarts and structured initializations
    
    from scipy.optimize import minimize
    
    best_solution = None
    
    # We will try ranks from 59 down to 40. 
    # We prioritize ranks close to 60 first as they are easier to find.
    # If we find one, we try to go lower.
    
    # To speed up, we can use the schoolbook decomposition as a starting point
    # and zero out some columns, then optimize.
    
    for R in range(59, 39, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        
        # Number of restarts depends on time remaining
        # For R close to 60, fewer restarts needed. For lower R, more.
        num_restarts = 10 if R > 55 else 20
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + restart * 1000 + R)
            
            # Initialization: Start with schoolbook, truncate to R columns, add noise
            # This helps if the optimal solution is "close" to schoolbook in some metric
            U_flat = np.zeros((n*m, R))
            V_flat = np.zeros((m*p, R))
            W_flat = np.zeros((p*n, R))
            
            # Copy first R columns of schoolbook if R <= 60
            if R <= 60:
                U_flat[:, :R] = U_sb[:, :R]
                V_flat[:, :R] = V_sb[:, :R]
                W_flat[:, :R] = W_sb[:, :R]
            
            # Add random perturbation to escape local minima
            # Scale perturbation based on R
            noise_scale = 0.1
            U_flat += np.random.randn(n*m, R) * noise_scale
            V_flat += np.random.randn(m*p, R) * noise_scale
            W_flat += np.random.randn(p*n, R) * noise_scale
            
            params = np.concatenate([U_flat.flatten(), V_flat.flatten(), W_flat.flatten()])
            
            try:
                result = minimize(
                    lambda p: objective(p, R),
                    params,
                    jac=lambda p: objective_grad(p, R)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                
                if result.fun < 1e-6: # Small error threshold
                    U_mat = result.x[:n*m*R].reshape(n*m, R)
                    V_mat = result.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                    W_mat = result.x[n*m*R+m*p*R:].reshape(p*n, R)
                    
                    # Try to round to half-integers
                    # The problem asks for integer/half-integer entries
                    U_rounded = np.round(U_mat * 2) / 2
                    V_rounded = np.round(V_mat * 2) / 2
                    W_rounded = np.round(W_mat * 2) / 2
                    
                    is_valid, max_diff = verify_decomposition(U_rounded, V_rounded, W_rounded)
                    if is_valid:
                        best_solution = (R, U_rounded.copy(), V_rounded.copy(), W_rounded.copy())
                        best_rank = R
                        found = True
                        break
                    
                    # Also check if the unrounded solution is "close enough" to be considered valid 
                    # (though the task requires exact decomposition, so this might not count 
                    # unless the evaluator has a tolerance. The task says "exactly decompose", 
                    # so we must stick to the rounded check or find an exact float solution 
                    # that represents integers exactly. But floats are imprecise. 
                    # Rounding is the best bet for "exact" in this context if the true solution is rational.)
                        
            except Exception:
                pass
        
        if found:
            # Found a solution for R, try to go lower
            continue
    
    # If we found a better solution, use it; otherwise use schoolbook
    if best_solution is not None:
        R, U_mat, V_mat, W_mat = best_solution
        best_rank = R
    else:
        U_mat, V_mat, W_mat = U_sb, V_sb, W_sb
    
    # Convert to lists as required by the interface
    U_list = U_mat.tolist()
    V_list = V_mat.tolist()
    W_list = W_mat.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
