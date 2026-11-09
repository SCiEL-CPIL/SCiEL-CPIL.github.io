# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] efficiently.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    s = support.astype(np.float64)
    
    if n == 1:
        return 0.0
    
    s_col = s.reshape(-1, 1)
    s_row = s.reshape(1, -1)
    pair_sums = s_col + s_row
    pair_probs = np.outer(w, w)
    
    sums_flat = pair_sums.flatten()
    probs_flat = pair_probs.flatten()
    
    sort_idx = np.argsort(sums_flat)
    sorted_sums = sums_flat[sort_idx]
    sorted_probs = probs_flat[sort_idx]
    cum_probs = np.cumsum(sorted_probs)
    
    if cum_probs[-1] < 1.0:
        if np.isclose(cum_probs[-1], 1.0, atol=1e-9):
            cum_probs[-1] = 1.0
    
    thresholds = 2.0 * s_row - s_col
    indices = np.searchsorted(sorted_sums, thresholds, side='left')
    cum_probs_with_zero = np.concatenate(([0.0], cum_probs))
    cdf_values = cum_probs_with_zero[indices]
    
    w_k = w.reshape(-1, 1)
    w_l = w.reshape(1, -1)
    
    P = np.sum(w_k * w_l * cdf_values)
    return float(P)

def _objective(x, n):
    """
    Objective function for scipy.optimize.minimize.
    x = [s_0, ..., s_{n-1}, log(w_0), ..., log(w_{n-1})]
    We minimize -P.
    Constraints: s_i >= 0, w_i > 0.
    """
    s = x[:n]
    log_w = x[n:]
    
    s_eff = np.maximum(s, 0.0)
    
    max_log_w = np.max(log_w)
    exp_w = np.exp(log_w - max_log_w)
    w_sum = np.sum(exp_w)
    if w_sum <= 1e-300:
        return 1e6
    w = exp_w / w_sum
    
    if np.any(~np.isfinite(s_eff)) or np.any(~np.isfinite(w)):
        return 1e6
    
    try:
        P = _compute_P_fast(s_eff, w)
    except Exception:
        return 1e6
        
    if not np.isfinite(P):
        return 1e6
        
    return -P

def _coordinate_ascent(s, w, max_iter=100, tol=1e-7):
    """
    Coordinate ascent: alternate between optimizing support positions and weights.
    """
    n = len(s)
    if n == 0:
        return s, w, 0.0
    
    w = np.maximum(w, 1e-15)
    w = w / np.sum(w)
    s = np.maximum(s, 0.0)
    
    best_P = _compute_P_fast(s, w)
    
    for iteration in range(max_iter):
        improved = False
        
        # Optimize each support position via golden section search
        for i in range(n):
            if w[i] < 1e-8:
                continue
            s_old = s[i]
            # Search range around current position
            lo = max(0.0, s_old - 1.0)
            hi = s_old + 1.0
            
            # Golden section search
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            a = lo
            b = hi
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            s_test_c = s.copy()
            s_test_c[i] = c
            P_c = _compute_P_fast(s_test_c, w)
            
            s_test_d = s.copy()
            s_test_d[i] = d
            P_d = _compute_P_fast(s_test_d, w)
            
            for _ in range(30):
                if P_c > P_d:
                    b = d
                    d = c
                    P_d = P_c
                    c = b - gr * (b - a)
                    s_test_c = s.copy()
                    s_test_c[i] = c
                    P_c = _compute_P_fast(s_test_c, w)
                else:
                    a = c
                    c = d
                    P_c = P_d
                    d = a + gr * (b - a)
                    s_test_d = s.copy()
                    s_test_d[i] = d
                    P_d = _compute_P_fast(s_test_d, w)
                
                if abs(b - a) < 1e-8:
                    break
            
            s_new = (a + b) / 2.0
            s_test = s.copy()
            s_test[i] = s_new
            P_new = _compute_P_fast(s_test, w)
            
            if P_new > best_P + tol:
                s[i] = s_new
                best_P = P_new
                improved = True
        
        # Optimize weights via gradient-free line search
        # Try adjusting weights proportionally
        for i in range(n):
            if w[i] < 1e-8:
                continue
            w_old = w[i]
            # Try increasing and decreasing weight
            for factor in [0.5, 0.8, 1.2, 2.0]:
                w_test = w.copy()
                w_test[i] = w_old * factor
                w_test = w_test / np.sum(w_test)
                P_test = _compute_P_fast(s, w_test)
                if P_test > best_P + tol:
                    w = w_test
                    best_P = P_test
                    improved = True
        
        if not improved:
            break
    
    # Final cleanup
    w[w < 1e-8] = 0.0
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return s, w, 0.0
    w = w / w_sum
    mask = w > 1e-6
    s_final = s[mask]
    w_final = w[mask]
    P_final = _compute_P_fast(s_final, w_final)
    return s_final, w_final, P_final

def _run_optimization(n, x0, t_start, time_limit):
    """Run L-BFGS-B optimization for a fixed n, followed by coordinate ascent."""
    deadline = t_start + time_limit
    
    bounds = [(0.0, 5.0)] * n + [(-8.0, 8.0)] * n
    
    result = None
    try:
        result = minimize(
            _objective, 
            x0, 
            args=(n,), 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30}
        )
    except Exception:
        pass
    
    if result is not None and result.fun < 1e5:
        x_opt = result.x
        s_opt = np.maximum(x_opt[:n], 0.0)
        log_w_opt = x_opt[n:]
        max_log_w = np.max(log_w_opt)
        exp_w = np.exp(log_w_opt - max_log_w)
        w_opt = exp_w / np.sum(exp_w)
        
        # Clean up tiny weights
        w_opt[w_opt < 1e-6] = 0.0
        if np.sum(w_opt) < 1e-9:
            return None, None, 0.0
        w_opt = w_opt / np.sum(w_opt)
        
        mask = w_opt > 1e-5
        s_final = s_opt[mask]
        w_final = w_opt[mask]
        
        if len(s_final) == 0:
            return None, None, 0.0
            
        P_final = _compute_P_fast(s_final, w_final)
        
        # Coordinate ascent refinement
        if time.time() < deadline - 5:
            s_ca, w_ca, P_ca = _coordinate_ascent(s_final, w_final, max_iter=50)
            if P_ca > P_final:
                s_final = s_ca
                w_final = w_ca
                P_final = P_ca
        
        return s_final, w_final, P_final
    else:
        return None, None, 0.0

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game.
    Strategy: Multi-start L-BFGS-B + coordinate ascent with adaptive n.
    """
    t_start = time.time()
    deadline = t_start + 590.0
    
    best_s = None
    best_w = None
    best_P = -1.0
    
    np.random.seed(42)
    
    # Focus on n=3,4,5,6 which seem most promising
    n_candidates = [3, 4, 5, 6]
    
    # Time allocation: ~120s per n value, 4 starts each
    for n in n_candidates:
        if time.time() > deadline - 60:
            break
        
        n_starts = 4
        
        for start_idx in range(n_starts):
            if time.time() > deadline - 20:
                break
            
            if start_idx == 0:
                s0 = np.linspace(0.01, 1.0, n)
            elif start_idx == 1:
                s0 = np.geomspace(0.01, 1.0, n)
            elif start_idx == 2:
                s0 = np.sort(np.random.uniform(0, 0.5, n))
            else:
                s0 = np.sort(np.random.uniform(0, 1.5, n))
                
            log_w0 = np.random.uniform(-3, 3, n)
            
            x0 = np.concatenate([s0, log_w0])
            
            time_limit = 25.0 if n <= 5 else 15.0
            
            s_opt, w_opt, P_opt = _run_optimization(n, x0, time.time(), time_limit)
            
            if s_opt is not None and P_opt > best_P:
                best_P = P_opt
                best_s = s_opt.copy()
                best_w = w_opt.copy()
                
    # If no valid solution found, return a baseline
    if best_s is None:
        s_base = np.array([0.0, 1.0])
        w_base = np.array([0.5, 0.5])
        return s_base, w_base
        
    # Final high-precision polish
    if time.time() < deadline - 30:
        n_best = len(best_s)
        s_curr = best_s.copy()
        w_curr = best_w.copy()
        
        w_curr = np.maximum(w_curr, 1e-10)
        w_curr = w_curr / np.sum(w_curr)
        
        # Extended coordinate ascent with more iterations
        s_polished, w_polished, P_polished = _coordinate_ascent(s_curr, w_curr, max_iter=200, tol=1e-9)
        
        if P_polished > best_P:
            best_P = P_polished
            best_s = s_polished
            best_w = w_polished
        
        # Try adding one more atom if time permits
        if time.time() < deadline - 15 and n_best < 8:
            # Add a new atom at a random position with small weight
            s_try = np.concatenate([best_s, [np.random.uniform(0, 2)]])
            w_try = np.concatenate([best_w, [0.05]])
            w_try = w_try / np.sum(w_try)
            
            s_new, w_new, P_new = _coordinate_ascent(s_try, w_try, max_iter=100)
            if P_new > best_P:
                best_P = P_new
                best_s = s_new
                best_w = w_new
    
    # Validation and formatting
    best_s = np.asarray(best_s, dtype=np.float64).flatten()
    best_w = np.asarray(best_w, dtype=np.float64).flatten()
    
    if len(best_s) == 0 or len(best_w) == 0:
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    if len(best_s) != len(best_w):
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    best_s = np.maximum(best_s, 0.0)
    best_w = np.maximum(best_w, 0.0)
    
    w_sum = np.sum(best_w)
    if w_sum <= 1e-9:
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    best_w = best_w / w_sum
    
    if not (np.all(np.isfinite(best_s)) and np.all(np.isfinite(best_w))):
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    sort_idx = np.argsort(best_s)
    best_s = best_s[sort_idx]
    best_w = best_w[sort_idx]
    
    return best_s, best_w

# EVOLVE-BLOCK-END
