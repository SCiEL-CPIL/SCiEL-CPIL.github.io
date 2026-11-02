import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Precompute valid mask
    valid_mask = np.zeros((nm, mp, pn), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                valid_mask[a, b, c] = True
                
    target_tensor = valid_mask.astype(float)
    
    best_rank = n * m * p
    best_U = np.zeros((nm, best_rank))
    best_V = np.zeros((mp, best_rank))
    best_W = np.zeros((pn, best_rank))
    
    # Initialize schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j, col] = 1
                best_V[j*p+k, col] = 1
                best_W[k*n+i, col] = 1
                col += 1
                
    start_time = time.time()
    deadline = 140
    
    # Try ranks from 12 up to schoolbook
    for R in range(12, best_rank + 1):
        if time.time() > deadline - 5:
            break
            
        found = False
        # Try a few seeds
        for seed in range(10):
            if time.time() > deadline - 5:
                break
                
            rng = np.random.RandomState(seed)
            # Random initialization
            size_U = nm * R
            size_V = mp * R
            size_W = pn * R
            params = rng.randn(size_U + size_V + size_W) * 0.1
            
            def obj(params):
                U = params[:size_U].reshape(nm, R)
                V = params[size_U:size_U+size_V].reshape(mp, R)
                W = params[size_U+size_V:].reshape(pn, R)
                S = np.einsum('ar,br,cr->abc', U, V, W)
                return np.sum((S - target_tensor)**2)
            
            try:
                res = minimize(obj, params, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
            except Exception:
                continue
                
            if res.fun < 1e-8:
                U = res.x[:size_U].reshape(nm, R)
                V = res.x[size_U:size_U+size_V].reshape(mp, R)
                W = res.x[size_U+size_V:].reshape(pn, R)
                
                # Round to nearest 0.5?
                # Check if rounded version is also valid
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                S_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                if np.sum((S_r - target_tensor)**2) < 1e-8:
                    U, V, W = U_r, V_r, W_r
                    
                # Final check
                S_final = np.einsum('ar,br,cr->abc', U, V, W)
                if np.sum((S_final - target_tensor)**2) < 1e-8:
                    best_rank = R
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break
        
        if found:
            # If we found a solution, we can stop if we want the minimal rank
            # But maybe a lower rank exists? We are iterating up, so 12 is lowest.
            # If 12 works, we return 12.
            break
            
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
