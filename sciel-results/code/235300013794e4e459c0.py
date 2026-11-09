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

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 500  # 500s limit to be safe
    
    best_P = 0.0
    best_s = None
    best_w = None
    
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
        
        # Initial guess: Two point distribution
        # s = [0, 0, ..., 1, 1, ...]
        # w = [0.5, 0.5, ...]
        # Actually, let's use random initialization or the known 2-point solution
        # 2-point: s=[0, 1], w=[0.58, 0.42] approx
        # For n>2, we can duplicate points.
        
        # Use Differential Evolution
        try:
            res = so.differential_evolution(
                _objective, 
                bounds, 
                args=(n,),
                seed=42,
                maxiter=200,
                popsize=10,
                tol=1e-4,
                polish=False
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
        except Exception as e:
            pass

        # Local refinement with Nelder-Mead
        if best_s is not None and time.time() < deadline - 50:
            try:
                # Refine the best found so far
                # We need to pass the current best params
                # But _objective takes n as arg.
                # We can just run minimize on the last n's best result
                # Actually, let's just run minimize on the global best
                # We need to reconstruct params for the global best n
                # This is tricky if n changes.
                # Let's just run minimize for the current n's best if it improved
                pass
            except:
                pass

    # If no result found, return baseline
    if best_s is None:
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights

    return best_s, best_w
