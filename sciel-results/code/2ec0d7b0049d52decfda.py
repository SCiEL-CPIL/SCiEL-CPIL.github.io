import numpy as np
import time
import random
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    start_time = time.time()
    deadline = start_time + 110.0

    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try ranks from 37 upwards
    # 37 is the sum of Laderman(23) + Rank14(14)
    target_ranks = [37, 38, 39, 40, 41, 42, 43, 44]
    
    for target_rank in target_ranks:
        if target_rank >= rank_sb:
            continue
        if time.time() > deadline - 15:
            break
            
        # Multiple restarts
        max_restarts = 10
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 15:
                break
            try:
                result = search_rank_scipy(n, m, p, target_rank, deadline)
                if result is not None:
                    return result
            except Exception:
                pass

    return best_result

def search_rank_scipy(n, m, p, rank, deadline):
    """
    Uses scipy.optimize.least_squares to minimize the Frobenius norm of the tensor decomposition error.
    """
    # Random initialization
    # Use small values to avoid large gradients initially?
    # Or standard normal?
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Flatten for optimizer
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def residuals(x):
        if time.time() > deadline - 10:
            raise TimeoutError()
            
        U_flat = x[:n*m*rank]
        V_flat = x[n*m*rank : n*m*rank + m*p*rank]
        W_flat = x[n*m*rank + m*p*rank :]
        
        U = U_flat.reshape(n, m, rank)
        V = V_flat.reshape(m, p, rank)
        W = W_flat.reshape(p, n, rank)
        
        # Compute T[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        
        return (T - 1.0).flatten()

    try:
        # Use 'trf' (Trust Region Reflective) which is robust for large problems
        # max_nfev: max number of function evaluations
        # We have ~1400 variables, 45 residuals.
        # It should be fast per step, but many steps might be needed.
        sol = least_squares(
            residuals, 
            x0, 
            method='trf',
            max_nfev=2000,
            x_scale='jac'
        )
    except TimeoutError:
        return None
    except Exception:
        return None

    if not sol.success and sol.status != 0:
        # Check if it's close enough regardless of success flag
        pass

    # Check error
    final_x = sol.x
    U_final = final_x[:n*m*rank].reshape(n, m, rank)
    V_final = final_x[n*m*rank : n*m*rank + m*p*rank].reshape(m, p, rank)
    W_final = final_x[n*m*rank + m*p*rank :].reshape(p, n, rank)
    
    T_final = np.einsum('ijr,jkr,kir->ijk', U_final, V_final, W_final)
    error = np.sum((T_final - 1.0)**2)
    
    if error < 1e-6:
        # Snap to half-integers
        U_snap = np.round(U_final * 2) / 2
        V_snap = np.round(V_final * 2) / 2
        W_snap = np.round(W_final * 2) / 2
        
        T_snap = np.einsum('ijr,jkr,kir->ijk', U_snap, V_snap, W_snap)
        snap_error = np.sum((T_snap - 1.0)**2)
        
        if snap_error < 1e-9:
            return {"U": U_snap.flatten().tolist(), 
                    "V": V_snap.flatten().tolist(), 
                    "W": W_snap.flatten().tolist()}
            
    return None
