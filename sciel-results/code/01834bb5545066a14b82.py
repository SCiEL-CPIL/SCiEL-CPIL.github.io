import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

def check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Checks if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    R = U.shape[1]
    U_3d = U.reshape(n, m, R)
    V_3d = V.reshape(m, p, R)
    W_3d = W.reshape(p, n, R)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=tol):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # 1. Construct Schoolbook
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_result = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    best_rank = rank_schoolbook
    
    deadline = time.time() + 120  # 120s safety margin
    
    # Try to find lower rank decompositions
    # Strategy: For target rank R, initialize by taking random linear combinations
    # of the schoolbook columns, then optimize.
    
    rng = np.random.default_rng(42)
    
    for target_rank in range(rank_schoolbook - 1, max(rank_schoolbook - 8, 45), -1):
        if time.time() > deadline:
            break
            
        found = False
        
        # Multiple random restarts
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize: take random linear combinations of schoolbook columns
            # Each new column is a weighted sum of schoolbook columns
            # We need the weighted sum to still produce valid tensor contributions
            
            # Simple init: copy first target_rank columns, then perturb
            U_init = U_sb[:, :target_rank].copy()
            V_init = V_sb[:, :target_rank].copy()
            W_init = W_sb[:, :target_rank].copy()
            
            # Add small random perturbations to help escape local minima
            if restart > 0:
                scale = 0.1 * (restart / max_restarts)
                U_init += rng.normal(0, scale, U_init.shape)
                V_init += rng.normal(0, scale, V_init.shape)
                W_init += rng.normal(0, scale, W_init.shape)
            
            def objective(params):
                U_flat = params[:n*m*target_rank]
                V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
                W_flat = params[n*m*target_rank + m*p*target_rank :]
                
                U_curr = U_flat.reshape(n, m, target_rank)
                V_curr = V_flat.reshape(m, p, target_rank)
                W_curr = W_flat.reshape(p, n, target_rank)
                
                T = np.einsum('ijr,jkr,kir->ijk', U_curr, V_curr, W_curr)
                error = np.sum((T - 1.0)**2)
                return error
            
            def jac(params):
                U_flat = params[:n*m*target_rank]
                V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
                W_flat = params[n*m*target_rank + m*p*target_rank :]
                
                U_curr = U_flat.reshape(n, m, target_rank)
                V_curr = V_flat.reshape(m, p, target_rank)
                W_curr = W_flat.reshape(p, n, target_rank)
                
                T = np.einsum('ijr,jkr,kir->ijk', U_curr, V_curr, W_curr)
                err = T - 1.0
                
                # d(error)/d(U[i,j,r]) = 2 * err[i,j,k] * V[j,k,r] * W[k,i,r] summed over k
                dU = 2.0 * np.einsum('ijk,jkr,kir->ijr', err, V_curr, W_curr)
                dV = 2.0 * np.einsum('ijk,ijr,kir->jkr', err, U_curr, W_curr)
                dW = 2.0 * np.einsum('ijk,ijr,jkr->kir', err, U_curr, V_curr)
                
                return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            
            initial_params = np.concatenate([
                U_init.flatten(),
                V_init.flatten(),
                W_init.flatten()
            ])
            
            try:
                res = minimize(objective, initial_params, jac=jac, method='L-BFGS-B', 
                             options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
                
                if res.fun < 1e-8:
                    U_sol = res.x[:n*m*target_rank].reshape(n, m, target_rank)
                    V_sol = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m, p, target_rank)
                    W_sol = res.x[n*m*target_rank + m*p*target_rank :].reshape(p, n, target_rank)
                    
                    # Round to nearest 0.5
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    if check_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-9):
                        best_rank = target_rank
                        best_result = {
                            "U": U_rounded.reshape(n*m, target_rank).tolist(),
                            "V": V_rounded.reshape(m*p, target_rank).tolist(),
                            "W": W_rounded.reshape(p*n, target_rank).tolist()
                        }
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
    
    return best_result
