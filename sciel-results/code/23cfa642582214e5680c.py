# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_start_time = time.time()
_DEADLINE = 590.0  # Safety margin: 630s limit, leave 40s buffer

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] efficiently using sorted pair sums.
    P = sum_{k,l} w_k w_l * Prob[X1+X2 < 2*s_l - s_k]
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    s = support.astype(np.float64)
    
    # Generate all pair sums and their probabilities
    s_col = s.reshape(-1, 1)
    s_row = s.reshape(1, -1)
    pair_sums = s_col + s_row  # (n, n)
    pair_probs = np.outer(w, w) # (n, n)
    
    # Flatten
    sums_flat = pair_sums.flatten() # (n^2,)
    probs_flat = pair_probs.flatten() # (n^2,)
    
    # Sort sums and compute cumulative probabilities
    sort_idx = np.argsort(sums_flat)
    sorted_sums = sums_flat[sort_idx]
    sorted_probs = probs_flat[sort_idx]
    cum_probs = np.cumsum(sorted_probs)
    
    # Ensure cum_probs ends at 1.0 (floating point safety)
    # If the last element is very close to 1, set it to 1 to avoid issues
    if abs(cum_probs[-1] - 1.0) < 1e-12:
        cum_probs[-1] = 1.0
    
    # Compute P = sum_{k,l} w_k w_l * CDF(2*s_l - s_k)
    # thresholds[k,l] = 2*s_l - s_k
    # We want sum of probs for pair_sums < threshold
    thresholds = 2.0 * s_row - s_col  # (n, n) where row index is k, col index is l?
    # Let's verify indices.
    # P = sum_{k} w_k sum_{l} w_l G(2*s_l - s_k)
    # Let's build thresholds matrix T where T[i,j] = 2*s_j - s_i
    # Then P = sum_{i,j} w_i w_j G(T[i,j])
    thresholds = 2.0 * s[None, :] - s[:, None] # shape (n, n), [k, l]
    
    # searchsorted(sorted_sums, thresholds, side='left') gives index of first >= threshold
    # We want sum of probs < threshold, so we take cum_probs[index-1] or 0 if index==0
    # np.searchsorted with side='left' returns the insertion index.
    # If we want sum of elements strictly less than T, we use the index returned by side='left'.
    # The sum is cum_probs[index-1] if index > 0 else 0.
    # We can use a trick: prepend 0.0 to cum_probs.
    cum_probs_with_zero = np.concatenate(([0.0], cum_probs))
    
    indices = np.searchsorted(sorted_sums, thresholds, side='left')
    cdf_values = cum_probs_with_zero[indices]
    
    # Sum over k, l with weights w_k w_l
    w_k = w.reshape(-1, 1)
    w_l = w.reshape(1, -1)
    
    P = np.sum(w_k * w_l * cdf_values)
    
    return float(P)

def _objective(params, n):
    """
    Objective to minimize: -P.
    params = [s_0,...,s_{n-1}, w_0,...,w_{n-1}]
    """
    s = params[:n]
    w = params[n:]
    
    # Enforce constraints softly or hard?
    # Hard clipping is better for stability in minimize
    s = np.maximum(s, 0.0)
    w = np.maximum(w, 1e-9) # Ensure positive weights for log/validity if needed, but here just sum
    
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return 1e10
        
    P = _compute_P_fast(s, w)
    return -P

def _normalize_and_check(s, w):
    """
    Normalize and check validity. Returns (s, w) or (None, None).
    """
    if s is None or w is None:
        return None, None
    if len(s) != len(w) or len(s) < 1:
        return None, None
        
    s = np.asarray(s, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    
    # Check for NaN/Inf
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(w)):
        return None, None
        
    # Clip support to >= 0
    s = np.maximum(s, 0.0)
    
    # Clip weights to >= 0
    w = np.maximum(w, 0.0)
    
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return None, None
        
    # Normalize weights
    w = w / w_sum
    
    # Sort by support to be tidy (not strictly required for P, but good for consistency)
    sort_idx = np.argsort(s)
    s = s[sort_idx]
    w = w[sort_idx]
    
    return s, w

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    best_support = None
    best_weights = None
    best_P = -1.0
    
    def update_best(s, w):
        nonlocal best_support, best_weights, best_P
        s_norm, w_norm = _normalize_and_check(s, w)
        if s_norm is None:
            return
        P = _compute_P_fast(s_norm, w_norm)
        if P > best_P:
            best_P = P
            best_support = s_norm.copy()
            best_weights = w_norm.copy()

    # 1. Initialize with known good candidates
    # The 2-point analytical optimum
    p_opt = (15 - np.sqrt(33)) / 16.0
    s2 = np.array([0.0, 1.0])
    w2 = np.array([p_opt, 1.0 - p_opt])
    update_best(s2, w2)
    
    # Geometric initializer
    s_geo = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    w_geo = np.array([0.5, 0.25, 0.125, 0.0625, 0.0625])
    update_best(s_geo, w_geo)
    
    # Random initializers
    np.random.seed(42)
    for _ in range(5):
        n_init = np.random.randint(3, 8)
        s_rand = np.sort(np.random.exponential(1.0, size=n_init))
        w_rand = np.random.dirichlet(np.ones(n_init))
        update_best(s_rand, w_rand)

    # 2. Local Optimization Loop
    # Try various n values
    max_n = 12
    for n in range(2, max_n + 1):
        if _time_left() < 60:
            break
            
        # Multiple random restarts for each n
        num_restarts = 15
        for restart in range(num_restarts):
            if _time_left() < 30:
                break
                
            # Generate initial params
            if restart == 0 and n == 2:
                # Use the analytical 2-point solution
                s0 = np.array([0.0, 1.0])
                w0 = np.array([p_opt, 1.0 - p_opt])
            elif restart == 1 and n >= 3:
                # Use geometric-ish start
                s0 = np.sort(np.linspace(0, 1, n) + np.random.normal(0, 0.1, n))
                w0 = np.array([0.5 ** i for i in range(n)])
                w0 = w0 / w0.sum()
            else:
                # Random exponential support, Dirichlet weights
                np.random.seed(1000 * n + restart)
                s0 = np.sort(np.random.exponential(1.0, size=n))
                w0 = np.random.dirichlet(np.ones(n))
            
            params0 = np.concatenate([s0, w0])
            
            # Powell Optimization
            try:
                res = minimize(_objective, params0, args=(n,), method='Powell',
                              options={'maxiter': 1000, 'maxfev': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
                if res.fun < -best_P:
                    s_opt = res.x[:n]
                    w_opt = res.x[n:]
                    update_best(s_opt, w_opt)
            except Exception:
                pass
            
            # Nelder-Mead Polish if Powell found something good
            if _time_left() < 30:
                continue
                
            try:
                # Use the best result from Powell as start for Nelder-Mead
                # Or just use params0 if Powell failed? No, use res.x if available.
                start_params = res.x if hasattr(res, 'x') else params0
                res_nm = minimize(_objective, start_params, args=(n,), method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
                if res_nm.fun < -best_P:
                    s_opt = res_nm.x[:n]
                    w_opt = res_nm.x[n:]
                    update_best(s_opt, w_opt)
            except Exception:
                pass

    # 3. Differential Evolution for Global Search (if time permits)
    # DE is slower but better for global exploration.
    if _time_left() > 120:
        for n in [3, 4, 5, 6]:
            if _time_left() < 60:
                break
            try:
                bounds = [(0, 10.0)] * n + [(1e-6, 1.0)] * n
                # Use seed for reproducibility
                de_res = differential_evolution(_objective, bounds, args=(n,),
                                                maxiter=50, popsize=20, seed=42+n,
                                                tol=1e-8, polish=True, init='sobol')
                if de_res.fun < -best_P:
                    s_opt = de_res.x[:n]
                    w_opt = de_res.x[n:]
                    update_best(s_opt, w_opt)
            except Exception:
                pass

    # 4. Final Polish on the absolute best found
    if best_support is not None and _time_left() > 10:
        n = len(best_support)
        params0 = np.concatenate([best_support, best_weights])
        try:
            res_final = minimize(_objective, params0, args=(n,), method='Powell',
                              options={'maxiter': 2000, 'maxfev': 10000, 'xatol': 1e-14, 'fatol': 1e-14})
            if res_final.fun < -best_P:
                s_opt = res_final.x[:n]
                w_opt = res_final.x[n:]
                update_best(s_opt, w_opt)
        except Exception:
            pass

    # Ensure validity of the returned result
    if best_support is None:
        # Fallback to 2-point solution
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([p_opt, 1.0 - p_opt])
    else:
        # Final normalization and clipping
        best_support = np.maximum(best_support, 0.0)
        best_weights = np.maximum(best_weights, 0.0)
        w_sum = np.sum(best_weights)
        if w_sum <= 1e-9:
            best_support = np.array([0.0, 1.0])
            best_weights = np.array([p_opt, 1.0 - p_opt])
        else:
            best_weights = best_weights / w_sum

    return best_support.astype(np.float64), best_weights.astype(np.float64)

# EVOLVE-BLOCK-END
