import numpy as np
from scipy.optimize import least_squares
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n, m, p = 2, 2, 10
    nm, mp, pn = n*m, m*p, p*n  # 4, 20, 20
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i*m+k, k*p+j, j*n+i] = 1.0
    
    # Strassen rank-35 solution
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    rank35 = 35
    U35 = np.zeros((nm, rank35))
    V35 = np.zeros((mp, rank35))
    W35 = np.zeros((pn, rank35))
    
    col = 0
    for k in range(p // 2):
        for a_vec, b_vec, c_vec in strassen_terms:
            for idx in range(nm):
                U35[idx, col] = a_vec[idx]
            V35[0 * p + 2 * k, col] = b_vec[0]
            V35[0 * p + 2 * k + 1, col] = b_vec[1]
            V35[1 * p + 2 * k, col] = b_vec[2]
            V35[1 * p + 2 * k + 1, col] = b_vec[3]
            W35[2 * k * n + 0, col] = c_vec[0]
            W35[2 * k * n + 1, col] = c_vec[1]
            W35[(2 * k + 1) * n + 0, col] = c_vec[2]
            W35[(2 * k + 1) * n + 1, col] = c_vec[3]
            col += 1
    
    # Try to find lower rank
    best_result = (U35, V35, W35)
    
    for target_rank in [34, 33, 32]:
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            
            # Initialize: take first target_rank columns of rank-35 solution
            U = U35[:, :target_rank].copy()
            V = V35[:, :target_rank].copy()
            W = W35[:, :target_rank].copy()
            
            # Add small random perturbation
            rng = np.random.RandomState(42 + attempt)
            U += 0.1 * rng.randn(nm, target_rank)
            V += 0.1 * rng.randn(mp, target_rank)
            W += 0.1 * rng.randn(pn, target_rank)
            
            # Solve using least squares
            def residuals(params):
                U_flat = params[:nm*target_rank]
                V_flat = params[nm*target_rank:nm*target_rank + mp*target_rank]
                W_flat = params[nm*target_rank + mp*target_rank:]
                U = U_flat.reshape(nm, target_rank)
                V = V_flat.reshape(mp, target_rank)
                W = W_flat.reshape(pn, target_rank)
                
                # Compute tensor decomposition
                # T[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s]
                res = np.zeros((nm, mp, pn))
                for s in range(target_rank):
                    res += np.einsum('a,b,c->abc', U[:, s], V[:, s], W[:, s])
                return (res - T).ravel()
            
            x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
            
            try:
                result = least_squares(residuals, x0, max_nfev=2000, ftol=1e-12, xtol=1e-12)
                if result.cost < 1e-8:
                    U_sol = result.x[:nm*target_rank].reshape(nm, target_rank)
                    V_sol = result.x[nm*target_rank:nm*target_rank + mp*target_rank].reshape(mp, target_rank)
                    W_sol = result.x[nm*target_rank + mp*target_rank:].reshape(pn, target_rank)
                    best_result = (U_sol, V_sol, W_sol)
                    found = True
                    break
            except Exception:
                pass
        
        if found:
            break
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
