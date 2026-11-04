import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    # Set a conservative deadline to ensure we return within 120s
    # Leaving 10s buffer for return serialization and evaluator overhead
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 45
    
    # Build schoolbook decomposition
    def build_schoolbook():
        U = np.zeros((n*m, schoolbook_rank))
        V = np.zeros((m*p, schoolbook_rank))
        W = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    # Verify a decomposition
    def verify(U, V, W, tol=1e-9):
        # Rec[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want Rec[i, j, k] == 1 for all i, j, k
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        # einsum indices: i j r, j k r, k i r -> i j k
        rec = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
        
        return np.allclose(rec, 1.0, atol=tol)

    # Get the schoolbook solution as the best fallback
    best_U, best_V, best_W = build_schoolbook()
    
    # Objective function for optimization
    def objective_and_grad(x, R):
        U_flat = x[:n*m*R].reshape(n*m, R)
        V_flat = x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W_flat = x[n*m*R + m*p*R:].reshape(p*n, R)
        
        U_reshaped = U_flat.reshape(n, m, R)
        V_reshaped = V_flat.reshape(m, p, R)
        W_reshaped = W_flat.reshape(p, n, R)
        
        # rec[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        rec = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
        
        error_vec = rec - 1.0
        error = np.sum(error_vec**2)
        
        # Gradients
        grad_U = 2.0 * np.einsum('ijk,jkr,kir->ijr', error_vec, V_reshaped, W_reshaped, optimize=True)
        grad_V = 2.0 * np.einsum('ijk,ijr,kir->jkr', error_vec, U_reshaped, W_reshaped, optimize=True)
        grad_W = 2.0 * np.einsum('ijk,ijr,jkr->kir', error_vec, U_reshaped, V_reshaped, optimize=True)
        
        return error, np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    # Try to find a lower rank decomposition
    # We search from 44 down to 42 (reduced range for safety)
    # Strategy: Initialize with the first R terms of the schoolbook solution.
    
    # Limit search to a small number of attempts to prevent timeout
    # We only try Rank 44 and 43, with 2 restarts each.
    # This is a refinement of the "search" family to be "safe search".
    
    for R in range(44, 41, -1):
        if time.time() > deadline - 30:  # Leave 30s for fallback and return
            break
            
        # Multiple restarts with different subsets of schoolbook terms
        for restart in range(2):
            if time.time() > deadline - 10:
                break
            
            # Select a random subset of R indices from the 45 schoolbook terms
            indices = np.random.choice(schoolbook_rank, R, replace=False)
            if restart == 0:
                indices = np.arange(R) # First R terms
            
            U_init = best_U[:, indices]
            V_init = best_V[:, indices]
            W_init = best_W[:, indices]
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                from scipy.optimize import minimize
                
                def obj(x):
                    e, _ = objective_and_grad(x, R)
                    return e
                
                def jac(x):
                    _, g = objective_and_grad(x, R)
                    return g
                
                # Reduced maxiter to keep it fast
                res = minimize(obj, x0, jac=jac, method='L-BFGS-B', 
                               options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6})
                
                if res.fun < 1e-6:
                    U_sol = res.x[:n*m*R].reshape(n*m, R)
                    V_sol = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                    W_sol = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                    
                    # Try to round to half-integers
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    if verify(U_rounded, V_rounded, W_rounded, tol=1e-9):
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                            
            except Exception:
                pass
    
    # Return schoolbook if no better found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
