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
    
    # Precompute thresholds: thresholds[k, l] = 2*s[l] - s[k]
    thresholds = 2.0 * s[None, :] - s[:, None]
    
    # Find indices of first element >= threshold (i.e., count of elements < threshold)
    indices = np.searchsorted(sorted_sums, thresholds, side='left')
    
    # Calculate sum of probabilities for elements < threshold
    sum_probs = np.zeros_like(indices, dtype=np.float64)
    mask_pos = indices > 0
    safe_indices = np.where(mask_pos, indices - 1, 0)
    safe_indices = np.clip(safe_indices, 0, len(cum_probs)-1)
    
    sum_probs[mask_pos] = cum_probs[safe_indices[mask_pos]]
    
    # P = sum_{k,l} p[k] p[l] sum_probs[k,l]
    P = np.sum(p[:, None] * p[None, :] * sum_probs)
    
    return P

def _optimize_weights_for_support(s, z0=None, maxiter=2000):
    """Optimize weights (via softmax) for a fixed support."""
    n = len(s)
    if z0 is None:
        z0 = np.zeros(n)
    
    def objective_softmax(z):
        exp_z = np.exp(z - np.max(z))
        w = exp_z / exp_z.sum()
        return -_compute_P_fast(s, w)
    
    try:
        res = minimize(objective_softmax, z0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        z_opt = res.x
        exp_z = np.exp(z_opt - np.max(z_opt))
        w_opt = exp_z / exp_z.sum()
        P_opt = -res.fun
        return s, w_opt, P_opt
    except Exception:
        return None, None, 0.0

def _joint_optimize(s_init, w_init, maxiter=2000):
    """Jointly optimize support and weights. Support parameterized as cumulative positive steps."""
    n = len(s_init)
    
    # Convert to step representation: s[0]=0, s[i] = s[i-1] + step[i-1]
    steps = np.diff(s_init)
    if np.any(steps <= 0):
        # Fallback: use absolute differences with minimum gap
        steps = np.maximum(np.abs(steps), 0.01)
    
    log_steps = np.log(steps)
    log_w = np.log(np.maximum(w_init, 1e-10))
    
    params = np.concatenate([log_steps, log_w])
    
    def objective(params):
        log_s = params[:n-1]
        log_w = params[n-1:]
        
        # Reconstruct support
        s = np.zeros(n)
        for i in range(1, n):
            s[i] = s[i-1] + np.exp(log_s[i-1])
        
        # Reconstruct weights via softmax
        max_log_w = np.max(log_w)
        exp_w = np.exp(log_w - max_log_w)
        w = exp_w / np.sum(exp_w)
        
        return -_compute_P_fast(s, w)
    
    try:
        res = minimize(objective, params, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        
        log_s = res.x[:n-1]
        log_w = res.x[n-1:]
        
        s = np.zeros(n)
        for i in range(1, n):
            s[i] = s[i-1] + np.exp(log_s[i-1])
        
        max_log_w = np.max(log_w)
        exp_w = np.exp(log_w - max_log_w)
        w = exp_w / np.sum(exp_w)
        
        P = -res.fun
        return s, w, P
    except Exception:
        return None, None, 0.0

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
    
    # 1. Analytical 2-point solution as a strong baseline
    q_opt = (15.0 - np.sqrt(33.0)) / 16.0
    s_2pt = np.array([0.0, 1.0])
    w_2pt = np.array([q_opt, 1.0 - q_opt])
    P_2pt = _compute_P_fast(s_2pt, w_2pt)
    if P_2pt > best_P:
        best_P = P_2pt
        best_support = s_2pt.copy()
        best_weights = w_2pt.copy()
    
    # 2. Explore diverse support structures with weight optimization
    supports_to_try = []
    
    # Geometric progressions
    for n in [3, 4, 5, 6, 8]:
        for base in [1.5, 2.0, 3.0, 5.0, 10.0]:
            s = np.array([base**i for i in range(n)], dtype=np.float64)
            supports_to_try.append(s)
    
    # Arithmetic progressions
    for n in [3, 4, 5, 6, 8]:
        for step in [0.5, 1.0, 2.0, 5.0]:
            s = np.arange(n, dtype=np.float64) * step
            supports_to_try.append(s)
    
    # Mixed: small cluster + large jumps
    for n in [5, 6, 8]:
        for k in range(2, min(n, 5)):
            s = np.zeros(n)
            s[:k] = np.arange(k, dtype=np.float64)
            for j in range(k, n):
                s[j] = 10.0 * (j - k + 1)
            supports_to_try.append(s)
    
    # Exponential-like: 0, 1, 2, 4, 8, ...
    for n in [4, 5, 6]:
        s = np.array([0.0] + [2.0**i for i in range(1, n)], dtype=np.float64)
        supports_to_try.append(s)
    
    # Try each support with weight optimization
    candidates = []
    for s in supports_to_try:
        if time.time() > deadline - 120.0:
            break
        
        # Use multiple initial guesses for weights
        n_s = len(s)
        initial_zs = [
            np.zeros(n_s),  # uniform
            np.log(np.ones(n_s) / n_s),  # same as above
            np.concatenate([np.log([1.5]), np.zeros(n_s-1)]),  # biased toward 0
            np.log(np.linspace(1.0, 0.3, n_s)),  # decreasing
        ]
        
        for z0 in initial_zs:
            if time.time() > deadline - 100.0:
                break
            s_opt, w_opt, P_opt = _optimize_weights_for_support(s, z0, maxiter=1500)
            if s_opt is not None and P_opt > best_P:
                best_P = P_opt
                best_support = s_opt.copy()
                best_weights = w_opt.copy()
                candidates.append((P_opt, s_opt.copy(), w_opt.copy()))
    
    # 3. Joint optimization on the best candidates from step 2
    if time.time() < deadline - 60.0:
        # Sort candidates by P and take top few
        candidates.sort(key=lambda x: -x[0])
        top_candidates = candidates[:5] if len(candidates) >= 5 else candidates
        
        for P_c, s_c, w_c in top_candidates:
            if time.time() > deadline - 30.0:
                break
            
            # Joint optimization starting from the weight-optimized solution
            s_j, w_j, P_j = _joint_optimize(s_c, w_c, maxiter=2000)
            if s_j is not None and P_j > best_P:
                best_P = P_j
                best_support = s_j.copy()
                best_weights = w_j.copy()
    
    # 4. Random support exploration in remaining time
    if time.time() < deadline - 30.0:
        np.random.seed(42)
        for _ in range(15):
            if time.time() > deadline - 10.0:
                break
            
            n = np.random.randint(3, 10)
            # Random sorted support with varying scales
            scale = np.random.uniform(1.0, 20.0)
            s_raw = np.sort(np.random.uniform(0, scale, n))
            # Ensure minimum gap
            for i in range(1, n):
                if s_raw[i] - s_raw[i-1] < 0.01:
                    s_raw[i] = s_raw[i-1] + 0.01
            
            # Quick weight optimization
            s_opt, w_opt, P_opt = _optimize_weights_for_support(s_raw, np.zeros(n), maxiter=1000)
            if s_opt is not None and P_opt > best_P:
                best_P = P_opt
                best_support = s_opt.copy()
                best_weights = w_opt.copy()
    
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
