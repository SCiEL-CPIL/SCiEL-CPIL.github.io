import numpy as np
import scipy.optimize as optimize
import time

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    timeout = 580 # safety margin before 630s limit
    
    N = 8
    
    def compute_volume(X):
        # X is array of shape (8,)
        # Use Gauss-Legendre quadrature for high accuracy
        c_vals, w_vals = np.polynomial.legendre.leggauss(100)
        # Map from [-1, 1] to [0, 1]
        c_vals = 0.5 * (c_vals + 1.0)
        w_vals = 0.5 * w_vals
        
        total_vol = 0.0
        for c, w in zip(c_vals, w_vals):
            # Intervals for each i
            starts = X + c * np.arange(N) / N
            ends = starts + 1.0 / N
            
            # Sort by start
            order = np.argsort(starts)
            s = starts[order]
            e = ends[order]
            
            # Merge intervals
            cur_s = s[0]
            cur_e = e[0]
            length = 0.0
            for k in range(1, N):
                if s[k] <= cur_e:
                    cur_e = max(cur_e, e[k])
                else:
                    length += cur_e - cur_s
                    cur_s = s[k]
                    cur_e = e[k]
            length += cur_e - cur_s
            
            total_vol += length * length * w
        
        return total_vol

    # Initial guess: linear shearing X_i = -i/8
    X0 = -np.arange(N) / N
    
    def obj(X_opt):
        X_full = np.concatenate(([0.0], X_opt))
        return compute_volume(X_full)
        
    X_init = X0[1:] - X0[0]
    
    X_best = np.concatenate(([0.0], X_init))
    best_val = obj(X_init)
    
    try:
        # Use Powell's method which is derivative-free and often better for this
        res = optimize.minimize(obj, X_init, method='Powell', 
                                options={'maxiter': 5000, 'xtol': 1e-8, 'ftol': 1e-10})
        
        if time.time() - start_time > timeout:
            raise TimeoutError()
            
        if res.fun < best_val:
            X_best = np.concatenate(([0.0], res.x))
            best_val = res.fun
            
    except Exception as e:
        pass
    
    # Try a second refinement with Nelder-Mead from the best point found
    try:
        if time.time() - start_time < timeout:
            X_init2 = X_best[1:]
            res2 = optimize.minimize(obj, X_init2, method='Nelder-Mead', 
                                    options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11})
            
            if time.time() - start_time > timeout:
                raise TimeoutError()
                
            if res2.fun < best_val:
                X_best = np.concatenate(([0.0], res2.x))
                best_val = res2.fun
                
    except Exception as e:
        pass
        
    # Construct positions
    positions = np.zeros((N*N, 2))
    for k in range(N*N):
        i = k // N
        j = k % N
        positions[k, 0] = X_best[i]
        positions[k, 1] = X_best[j]
        
    return positions
