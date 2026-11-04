import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    U_sch = np.zeros((n*m, schoolbook_rank))
    V_sch = np.zeros((m*p, schoolbook_rank))
    W_sch = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j, col] = 1
                V_sch[j*p+k, col] = 1
                W_sch[k*n+i, col] = 1
                col += 1
    
    best_result = (U_sch.tolist(), V_sch.tolist(), W_sch.tolist())
    best_rank = schoolbook_rank
    
    # Search for lower ranks
    # Lower bound is max(nm, mp, pn) = 15
    # Try from 15 up to schoolbook_rank
    for target_rank in range(15, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
            
        # If we found a better rank in a previous iteration, we might have broken?
        # The loop goes upwards. We want the MINIMAL rank.
        # So we should iterate from low to high.
        # If we find a valid rank, we can return it immediately if we are confident it's minimal?
        # No, we should check all lower ranks first.
        # Since we are iterating from 15 upwards, the first valid rank we find is the minimal one (among those we test).
        
        found = False
        max_restarts = 10 if target_rank < 30 else 30
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            U_init = np.random.randn(n*m, target_rank) * 0.5
            V_init = np.random.randn(m*p, target_rank) * 0.5
            W_init = np.random.randn(p*n, target_rank) * 0.5
            
            # Define objective and gradient
            def objective_and_grad(params):
                U_r = params[:n*m*target_rank].reshape(n*m, target_rank)
                V_r = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_r = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                # Compute T_approx
                # T_approx[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # We can do this with einsum
                T_approx = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                
                # Error
                diff = T - T_approx
                obj = np.sum(diff**2)
                
                # Gradient
                # dE/dU[a, r] = -2 * sum_{b,c} diff[a, b, c] V[b, r] W[c, r]
                # dE/dV[b, r] = -2 * sum_{a,c} diff[a, b, c] U[a, r] W[c, r]
                # dE/dW[c, r] = -2 * sum_{a,b} diff[a, b, c] U[a, r] V[b, r]
                
                grad_U = -2 * np.einsum('abc,br,cr->ar', diff, V_r, W_r)
                grad_V = -2 * np.einsum('abc,ar,cr->br', diff, U_r, W_r)
                grad_W = -2 * np.einsum('abc,ar,br->cr', diff, U_r, V_r)
                
                grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                
                return obj, grad
            
            params_init = np.concatenate([
                U_init.flatten(),
                V_init.flatten(),
                W_init.flatten()
            ])
            
            try:
                result = minimize(objective_and_grad, params_init, method='L-BFGS-B', 
                                 jac=True, options={'maxiter': 100, 'maxfun': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if result.fun < 1e-8:
                    U_r = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_r = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_r = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Try rounding to integers and half-integers
                    # First try integers
                    U_int = np.round(U_r)
                    V_int = np.round(V_r)
                    W_int = np.round(W_r)
                    T_approx_int = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
                    if np.allclose(T, T_approx_int, atol=1e-10):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                    
                    # Try half-integers
                    U_half = np.round(U_r * 2) / 2
                    V_half = np.round(V_r * 2) / 2
                    W_half = np.round(W_r * 2) / 2
                    T_approx_half = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
                    if np.allclose(T, T_approx_half, atol=1e-10):
                        return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
                        
                    # If not exact, maybe we can try to refine with higher precision?
                    # But for now, we accept if it's very close? No, must be exact.
                    
            except Exception:
                pass
        
        # If we found a valid rank, we can break because we are iterating upwards
        # But we didn't set 'found' to True if we returned.
        # If we didn't return, we continue to next rank.
        
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
# EVOLVE-BLOCK-END
