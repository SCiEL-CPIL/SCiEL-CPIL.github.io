import numpy as np
import scipy.optimize as so
import time

def _compute_P_fast(support, weights):
    n = len(support)
    if n == 0:
        return 0.0
    w_sum = np.sum(weights)
    if w_sum <= 1e-12:
        return 0.0
    w = weights / w_sum
    s = support
    
    S = s[:, None] + s[None, :]
    M = np.outer(w, w)
    
    S_flat = S.flatten()
    M_flat = M.flatten()
    
    sort_idx = np.argsort(S_flat)
    S_sorted = S_flat[sort_idx]
    M_sorted = M_flat[sort_idx]
    M_cumsum = np.cumsum(M_sorted)
    
    T = 2.0 * s[None, :] - s[:, None]
    T_flat = T.flatten()
    
    counts = np.searchsorted(S_sorted, T_flat, side='left')
    
    inner_sums = np.zeros_like(T_flat)
    mask = counts > 0
    inner_sums[mask] = M_cumsum[counts[mask] - 1]
    
    W = np.outer(w, w)
    
    P = np.sum(W * inner_sums.reshape(n, n))
    
    return P

def _objective(params, n):
    # params: [s_1, ..., s_{n-1}, w_0, ..., w_{n-1}]
    # s_0 is fixed to 0
    s = np.zeros(n)
    s[1:] = params[:n-1]
    
    w = params[n-1:]
    
    # Ensure w is non-negative (DE bounds handle this)
    # Normalize w
    w_sum = np.sum(w)
    if w_sum <= 0:
        return 0.0
    w = w / w_sum
    
    return -_compute_P_fast(s, w)

def _objective_for_minimize(params, n):
    """Wrapper for scipy minimize that unpacks and calls _objective."""
    return _objective(params, n)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 580  # 580s limit, 50s safety margin
    
    best_P = 0.0
    best_s = None
    best_w = None
    best_n = 0
    
    # Try different n
    for n in [4, 8, 16, 32]:
        if time.time() > deadline:
            break
        
        # Bounds
        # s_i in [0, 10]
        # w_i in [0, 1]
        bounds_s = [(0.0, 10.0)] * (n - 1)
        bounds_w = [(0.0, 1.0)] * n
        bounds = bounds_s + bounds_w
        
        # Use geometric decay as initial guess for better structure
        # s_i = i * scale, w_i proportional to exp(-i * decay)
        # This concentrates mass at small values while allowing tail
        
        # Initial guess with geometric structure
        s_init = np.zeros(n)
        for i in range(1, n):
            s_init[i] = i * 0.5  # spread out
        w_init = np.exp(-np.arange(n) * 0.3)  # geometric decay
        w_init = w_init / w_init.sum()
        
        # DE initial population: mix of geometric and random
        x0_list = [np.concatenate([s_init[1:], w_init])]
        
        # Add some random initial points
        rng = np.random.RandomState(42)
        for _ in range(5):
            s_rand = np.zeros(n)
            s_rand[1:] = rng.uniform(0, 10, n-1)
            w_rand = rng.dirichlet(np.ones(n))
            x0_list.append(np.concatenate([s_rand[1:], w_rand]))
        
        x0 = np.array(x0_list)
        
        # Use Differential Evolution with better initialization
        try:
            res = so.differential_evolution(
                _objective, 
                bounds, 
                args=(n,),
                seed=42,
                maxiter=300,
                popsize=15,
                tol=1e-5,
                polish=False,
                init=x0
            )
            
            if res.fun < 0:
                P = -res.fun
                if P > best_P:
                    best_P = P
                    params = res.x
                    s = np.zeros(n)
                    s[1:] = params[:n-1]
                    w = params[n-1:]
                    w_sum = np.sum(w)
                    if w_sum > 0:
                        w = w / w_sum
                    best_s = s
                    best_w = w
                    best_n = n
        except Exception as e:
            pass

        # Local refinement with Nelder-Mead if we found something
        if best_s is not None and best_n == n and time.time() < deadline - 80:
            try:
                # Reconstruct params for minimize
                s_curr = best_s
                w_curr = best_w
                params_curr = np.concatenate([s_curr[1:], w_curr])
                
                # Nelder-Mead refinement
                res_nm = so.minimize(
                    _objective_for_minimize,
                    params_curr,
                    args=(n,),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                
                if res_nm.fun < 0:
                    P_nm = -res_nm.fun
                    if P_nm > best_P:
                        best_P = P_nm
                        params_nm = res_nm.x
                        s_nm = np.zeros(n)
                        s_nm[1:] = params_nm[:n-1]
                        w_nm = params_nm[n-1:]
                        w_sum = np.sum(w_nm)
                        if w_sum > 0:
                            w_nm = w_nm / w_sum
                        best_s = s_nm
                        best_w = w_nm
            except Exception as e:
                pass
    
    # Final high-precision polish if we have time
    if best_s is not None and time.time() < deadline - 20:
        try:
            n_final = len(best_s)
            s_final = best_s.copy()
            w_final = best_w.copy()
            params_final = np.concatenate([s_final[1:], w_final])
            
            # Very tight Nelder-Mead
            res_polish = so.minimize(
                _objective_for_minimize,
                params_final,
                args=(n_final,),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if res_polish.fun < 0:
                P_polish = -res_polish.fun
                if P_polish > best_P:
                    best_P = P_polish
                    params_polish = res_polish.x
                    s_polish = np.zeros(n_final)
                    s_polish[1:] = params_polish[:n_final-1]
                    w_polish = params_polish[n_final-1:]
                    w_sum = np.sum(w_polish)
                    if w_sum > 0:
                        w_polish = w_polish / w_sum
                    best_s = s_polish
                    best_w = w_polish
        except Exception as e:
            pass

    # If no result found, return baseline
    if best_s is None:
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights

    return best_s, best_w
