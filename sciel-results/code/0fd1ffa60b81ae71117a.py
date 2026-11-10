# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_P_fast(support, weights):
    """
    Fast computation of P[X1+X2+X3 < 2*X4] using sorting and binary search.
    Complexity: O(n^2 log n)
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum()
    if total <= 1e-9:
        return 0.0
    p = w / total
    n = len(s)
    
    if n == 0:
        return 0.0

    # Pairwise sums and probabilities
    pair_sums = s[:, None] + s[None, :]
    pair_probs = p[:, None] * p[None, :]
    
    # Flatten for sorting
    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()
    
    # Sort sums and get corresponding probabilities
    sort_idx = np.argsort(flat_sums)
    sorted_sums = flat_sums[sort_idx]
    sorted_probs = flat_probs[sort_idx]
    
    # Cumulative sum of probabilities
    cum_probs = np.cumsum(sorted_probs)
    
    # Precompute thresholds
    thresholds = 2.0 * s[None, :] - s[:, None]
    
    # For each threshold, find the number of pair_sums strictly less than it
    indices = np.searchsorted(sorted_sums, thresholds, side='left')
    
    # Now we need the sum of sorted_probs up to indices.
    sum_probs = np.zeros_like(indices, dtype=np.float64)
    
    # Mask for indices > 0
    mask_pos = indices > 0
    safe_indices = np.where(mask_pos, indices - 1, 0)
    safe_indices = np.clip(safe_indices, 0, len(cum_probs)-1)
    
    sum_probs[mask_pos] = cum_probs[safe_indices[mask_pos]]
    
    # Now P = sum_{k,l} p[k] p[l] sum_probs[k,l]
    P = np.sum(p[:, None] * p[None, :] * sum_probs)
    
    return P

def _polish_weights(s, w_init, max_iter=5000, ftol=1e-16):
    """
    Polish weights for a fixed support using L-BFGS-B on the simplex.
    Uses softmax parameterization to stay on the simplex.
    """
    n = len(s)
    
    def objective_softmax(z):
        exp_z = np.exp(z - np.max(z))
        w = exp_z / exp_z.sum()
        P = _compute_P_fast(s, w)
        return -P
    
    # Start from current weights
    w = np.asarray(w_init, dtype=np.float64)
    w = np.clip(w, 1e-15, None)
    w = w / w.sum()
    z0 = np.log(w)
    
    try:
        res = minimize(objective_softmax, z0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-12})
        if res.fun < 0:  # Negative means P > 0
            z_opt = res.x
            exp_z = np.exp(z_opt - np.max(z_opt))
            w_opt = exp_z / exp_z.sum()
            return w_opt
    except Exception:
        pass
    
    return w_init

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 600.0 - 30.0  # 30s safety margin
    
    best_P = 0.0
    best_support = np.array([0.0, 1.0])
    best_weights = np.array([0.5, 0.5])
    
    def try_candidate(s, w, label=""):
        nonlocal best_P, best_support, best_weights
        s = np.asarray(s, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        if len(s) != len(w) or len(s) == 0:
            return
        if np.any(s < -1e-9) or np.any(w < -1e-9):
            return
        w = np.clip(w, 0, None)
        total = w.sum()
        if total <= 1e-9:
            return
        w = w / total
        
        P = _compute_P_fast(s, w)
        if P > best_P:
            best_P = P
            best_support = s.copy()
            best_weights = w.copy()
    
    # === Strategy 1: Two-point mixture (point mass at 0 + Pareto tail) ===
    # The key insight: with mass (1-alpha) at 0, P[X1+X2+X3=0] = (1-alpha)^3
    # which automatically satisfies X1+X2+X3 < 2*X4 for any X4 > 0.
    # The remaining alpha mass is distributed over Pareto quantiles to optimize
    # the non-zero sum cases.
    
    for n_tail in [2, 3, 4, 5, 6, 8, 10]:
        if time.time() > deadline - 120.0:
            break
        
        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            if time.time() > deadline - 60.0:
                break
            
            for pareto_alpha in [1.0, 1.5, 2.0, 3.0, 5.0]:
                if time.time() > deadline - 30.0:
                    break
                
                # Pareto quantiles: x_p = x_min * (1-p)^(-1/pareto_alpha)
                # Use p in [0.01, 0.99] to avoid extreme values
                p_vals = np.linspace(0.01, 0.99, n_tail)
                x_min = 1.0
                tail_support = x_min * (1 - p_vals) ** (-1.0 / pareto_alpha)
                
                # Full support: 0 at first position, then tail
                s = np.concatenate([[0.0], tail_support])
                
                # Initial weights: (1-alpha) at 0, alpha/n_tail for each tail point
                w = np.zeros(len(s))
                w[0] = 1.0 - alpha
                w[1:] = alpha / n_tail
                
                try_candidate(s, w, f"mixture_n={n_tail},alpha={alpha},pa={pareto_alpha}")
                
                # Polish weights
                if time.time() < deadline - 30.0:
                    w_polished = _polish_weights(s, w, max_iter=2000, ftol=1e-16)
                    try_candidate(s, w_polished, f"mixture_polished_n={n_tail},alpha={alpha},pa={pareto_alpha}")
    
    # === Strategy 2: Original geometric/arithmetic supports ===
    
    supports_to_try = []
    
    # Geometric
    for n in [3, 4, 5, 6, 8, 10]:
        for base in [1.5, 2.0, 3.0, 5.0]:
            s = np.array([base**i for i in range(n)])
            supports_to_try.append(s)
            
    # Arithmetic
    for n in [3, 4, 5, 6, 8, 10, 15]:
        for step in [1.0, 2.0, 0.5]:
            s = np.arange(n) * step
            supports_to_try.append(s)
            
    # Mixed: 0, 1, ..., k, then large jumps
    for n in [5, 6, 8]:
        for k in range(2, n):
            s = np.zeros(n)
            s[:k] = np.arange(k)
            for j in range(k, n):
                s[j] = 10.0 * (j - k + 1)
            supports_to_try.append(s)
            
    # Optimize weights for each support
    for s in supports_to_try:
        if time.time() > deadline - 60.0:
            break
            
        n = len(s)
        
        # Initial guess: uniform
        w0 = np.ones(n) / n
        
        def objective_softmax(z):
            exp_z = np.exp(z - np.max(z))
            w = exp_z / exp_z.sum()
            P = _compute_P_fast(s, w)
            return -P
        
        z0 = np.log(w0)
        
        try:
            res = minimize(objective_softmax, z0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.success or res.fun < -best_P:
                z_opt = res.x
                exp_z = np.exp(z_opt - np.max(z_opt))
                w_opt = exp_z / exp_z.sum()
                
                P_opt = -res.fun
                if P_opt > best_P:
                    best_P = P_opt
                    best_support = s.copy()
                    best_weights = w_opt.copy()
                    
        except Exception:
            pass
        
        # Weights-only polish
        if time.time() < deadline - 30.0:
            w_polished = _polish_weights(best_support, best_weights, max_iter=3000, ftol=1e-18)
            try_candidate(best_support, w_polished, "polish_original")

    # Final check: try a few random supports with optimization
    if time.time() < deadline - 120.0:
        np.random.seed(42)
        for _ in range(20):
            n = np.random.randint(3, 12)
            s_raw = np.sort(np.random.uniform(0, 20, n))
            s = s_raw
            z0 = np.zeros(n)
            
            def objective_softmax(z):
                exp_z = np.exp(z - np.max(z))
                w = exp_z / exp_z.sum()
                P = _compute_P_fast(s, w)
                return -P
                
            try:
                res = minimize(objective_softmax, z0, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-6})
                if res.fun < -best_P:
                    z_opt = res.x
                    exp_z = np.exp(z_opt - np.max(z_opt))
                    w_opt = exp_z / exp_z.sum()
                    P_opt = -res.fun
                    if P_opt > best_P:
                        best_P = P_opt
                        best_support = s.copy()
                        best_weights = w_opt.copy()
            except Exception:
                pass

    # Ensure validity
    if len(best_support) != len(best_weights):
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
    if np.any(best_support < -1e-9):
        best_support = np.abs(best_support)
        
    if np.any(best_weights < -1e-9):
        best_weights = np.abs(best_weights)
        
    total_w = best_weights.sum()
    if total_w <= 1e-9:
        best_weights = np.ones(len(best_support)) / len(best_support)
    else:
        best_weights = best_weights / total_w
        
    return best_support, best_weights

# EVOLVE-BLOCK-END
