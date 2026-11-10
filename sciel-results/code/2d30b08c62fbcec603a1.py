# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _compute_P_fast(support, weights):
    """
    Fast exact computation of P[X1+X2+X3 < 2*X4] using sorting and binary search.
    Complexity: O(n^2 log n).
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

    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()

    sort_idx = np.argsort(flat_sums, kind='stable')
    sorted_sums = flat_sums[sort_idx]
    sorted_probs = flat_probs[sort_idx]
    cum_probs = np.cumsum(sorted_probs)

    # thresholds[k, l] = 2*s[l] - s[k]
    thresholds = 2.0 * s[None, :] - s[:, None]

    # count of elements strictly < threshold
    indices = np.searchsorted(sorted_sums, thresholds, side='left')

    sum_probs = np.zeros_like(indices, dtype=np.float64)
    mask_pos = indices > 0
    safe_indices = np.where(mask_pos, indices - 1, 0)
    safe_indices = np.clip(safe_indices, 0, len(cum_probs) - 1)
    sum_probs[mask_pos] = cum_probs[safe_indices[mask_pos]]

    P = np.sum(p[:, None] * p[None, :] * sum_probs)
    return float(P)

def _compute_P_fast_cached(support, weights, cache_key=None):
    """
    Cached version: if support hasn't changed, reuse sorted pair sums.
    """
    if cache_key is not None:
        cached = _compute_P_fast_cached._cache
        if cache_key in cached:
            sorted_sums, sort_idx = cached[cache_key]
            s = np.asarray(support, dtype=np.float64)
            w = np.asarray(weights, dtype=np.float64)
            total = w.sum()
            if total <= 1e-9:
                return 0.0
            p = w / total
            n = len(s)
            if n == 0:
                return 0.0
            pair_probs = p[:, None] * p[None, :]
            flat_probs = pair_probs.flatten()
            sorted_probs = flat_probs[sort_idx]
            cum_probs = np.cumsum(sorted_probs)
            thresholds = 2.0 * s[None, :] - s[:, None]
            indices = np.searchsorted(sorted_sums, thresholds, side='left')
            sum_probs = np.zeros_like(indices, dtype=np.float64)
            mask_pos = indices > 0
            safe_indices = np.where(mask_pos, indices - 1, 0)
            safe_indices = np.clip(safe_indices, 0, len(cum_probs) - 1)
            sum_probs[mask_pos] = cum_probs[safe_indices[mask_pos]]
            P = np.sum(p[:, None] * p[None, :] * sum_probs)
            return float(P)
    return _compute_P_fast(support, weights)
_compute_P_fast_cached._cache = {}

def _optimize_weights_l_bfgs(s, w0, maxiter=1000):
    """
    Optimize weights for a fixed support using L-BFGS-B on log-weights.
    """
    n = len(s)
    log_w0 = np.log(np.maximum(w0, 1e-10))
    
    # Precompute sorted pair sums for this fixed support
    s_arr = np.asarray(s, dtype=np.float64)
    pair_sums = s_arr[:, None] + s_arr[None, :]
    flat_sums = pair_sums.flatten()
    sort_idx = np.argsort(flat_sums, kind='stable')
    sorted_sums = flat_sums[sort_idx]
    
    def objective(params):
        log_w = params
        max_log_w = np.max(log_w)
        exp_w = np.exp(log_w - max_log_w)
        w = exp_w / np.sum(exp_w)
        # Use cached sorted sums
        p = w
        pair_probs = p[:, None] * p[None, :]
        flat_probs = pair_probs.flatten()
        sorted_probs = flat_probs[sort_idx]
        cum_probs = np.cumsum(sorted_probs)
        thresholds = 2.0 * s_arr[None, :] - s_arr[:, None]
        indices = np.searchsorted(sorted_sums, thresholds, side='left')
        sum_probs = np.zeros_like(indices, dtype=np.float64)
        mask_pos = indices > 0
        safe_indices = np.where(mask_pos, indices - 1, 0)
        safe_indices = np.clip(safe_indices, 0, len(cum_probs) - 1)
        sum_probs[mask_pos] = cum_probs[safe_indices[mask_pos]]
        P_val = np.sum(p[:, None] * p[None, :] * sum_probs)
        return -float(P_val)
    
    try:
        res = minimize(objective, log_w0, method='L-BFGS-B', 
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
        log_w_opt = res.x
        max_log_w = np.max(log_w_opt)
        exp_w = np.exp(log_w_opt - max_log_w)
        w_opt = exp_w / np.sum(exp_w)
        P_opt = -res.fun
        return s, w_opt, P_opt
    except Exception:
        return None, None, 0.0

def _polish_weights(s, w0, maxiter=5000):
    """
    High-precision polish of weights for a fixed support using Nelder-Mead with tight tolerances.
    """
    n = len(s)
    w_safe = np.maximum(w0, 1e-15)
    z0 = np.log(w_safe)
    
    # Precompute sorted pair sums
    s_arr = np.asarray(s, dtype=np.float64)
    pair_sums = s_arr[:, None] + s_arr[None, :]
    flat_sums = pair_sums.flatten()
    sort_idx = np.argsort(flat_sums, kind='stable')
    sorted_sums = flat_sums[sort_idx]
    
    def objective_softmax(z):
        exp_z = np.exp(z - np.max(z))
        w = exp_z / exp_z.sum()
        p = w
        pair_probs = p[:, None] * p[None, :]
        flat_probs = pair_probs.flatten()
        sorted_probs = flat_probs[sort_idx]
        cum_probs = np.cumsum(sorted_probs)
        thresholds = 2.0 * s_arr[None, :] - s_arr[:, None]
        indices = np.searchsorted(sorted_sums, thresholds, side='left')
        sum_probs = np.zeros_like(indices, dtype=np.float64)
        mask_pos = indices > 0
        safe_indices = np.where(mask_pos, indices - 1, 0)
        safe_indices = np.clip(safe_indices, 0, len(cum_probs) - 1)
        sum_probs[mask_pos] = cum_probs[safe_indices[mask_pos]]
        P_val = np.sum(p[:, None] * p[None, :] * sum_probs)
        return -float(P_val)

    try:
        res = minimize(objective_softmax, z0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
        z_opt = res.x
        exp_z = np.exp(z_opt - np.max(z_opt))
        w_opt = exp_z / exp_z.sum()
        P_opt = -res.fun
        return w_opt, P_opt
    except Exception:
        return w0, _compute_P_fast(s, w0)

def _joint_optimize_l_bfgs(s_init, w_init, maxiter=1000):
    """
    Jointly optimize support and weights using L-BFGS-B on log-steps and log-weights.
    """
    n = len(s_init)
    steps = np.diff(s_init)
    if np.any(steps <= 0):
        steps = np.maximum(np.abs(steps), 0.01)
    log_steps = np.log(steps)
    log_w_init = np.log(np.maximum(w_init, 1e-10))
    params0 = np.concatenate([log_steps, log_w_init])
    
    def objective(params):
        log_s = params[:n - 1]
        log_w = params[n - 1:]
        s = np.zeros(n)
        s[1:] = np.cumsum(np.exp(log_s))
        
        max_log_w = np.max(log_w)
        exp_w = np.exp(log_w - max_log_w)
        w = exp_w / np.sum(exp_w)
        return -_compute_P_fast(s, w)
    
    try:
        res = minimize(objective, params0, method='L-BFGS-B', 
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
        log_s = res.x[:n - 1]
        log_w = res.x[n - 1:]
        s = np.zeros(n)
        s[1:] = np.cumsum(np.exp(log_s))
        
        max_log_w = np.max(log_w)
        exp_w = np.exp(log_w - max_log_w)
        w = exp_w / np.sum(exp_w)
        P = -res.fun
        return s, w, P
    except Exception:
        return None, None, 0.0

def _coordinate_descent_polish(s, w, max_iter=200, tol=1e-10):
    """
    Fast coordinate descent polish: alternate between optimizing weights (given s)
    and optimizing support (given w) using small perturbations.
    """
    n = len(s)
    s = np.array(s, dtype=np.float64)
    w = np.array(w, dtype=np.float64)
    P_curr = _compute_P_fast(s, w)
    
    for it in range(max_iter):
        # Optimize weights via L-BFGS-B (fast since support is fixed)
        s_opt, w_opt, P_opt = _optimize_weights_l_bfgs(s, w, maxiter=200)
        if s_opt is not None and P_opt > P_curr + tol:
            w = w_opt
            P_curr = P_opt
        
        # Try small perturbations on support
        improved = False
        for i in range(1, n):
            for delta in [0.01, 0.1, 1.0, -0.01, -0.1]:
                s_try = s.copy()
                s_try[i:] += delta
                if np.any(s_try < -1e-9):
                    continue
                if np.any(np.diff(s_try) < 1e-6):
                    continue
                P_try = _compute_P_fast(s_try, w)
                if P_try > P_curr + tol:
                    s = s_try
                    P_curr = P_try
                    improved = True
        if not improved and it > 10:
            break
    
    return s, w, P_curr

def _polish_joint(s_init, w_init, maxiter=5000):
    """
    High-precision joint polish: first coordinate descent, then Nelder-Mead on remaining degrees.
    """
    n = len(s_init)
    # Stage 1: Coordinate descent
    s_cd, w_cd, P_cd = _coordinate_descent_polish(s_init, w_init, max_iter=100, tol=1e-9)
    
    # Stage 2: Nelder-Mead polish on the result
    steps = np.diff(s_cd)
    if np.any(steps <= 0):
        steps = np.maximum(np.abs(steps), 0.001)
    log_steps = np.log(steps)
    log_w = np.log(np.maximum(w_cd, 1e-15))
    params = np.concatenate([log_steps, log_w])

    def objective(params):
        log_s = params[:n - 1]
        log_w = params[n - 1:]
        s = np.zeros(n)
        s[1:] = np.cumsum(np.exp(log_s))
        max_log_w = np.max(log_w)
        exp_w = np.exp(log_w - max_log_w)
        w = exp_w / np.sum(exp_w)
        return -_compute_P_fast(s, w)

    try:
        res = minimize(objective, params, method='Nelder-Mead',
                       options={'maxiter': min(maxiter, 2000), 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
        log_s = res.x[:n - 1]
        log_w = res.x[n - 1:]
        s = np.zeros(n)
        s[1:] = np.cumsum(np.exp(log_s))
        max_log_w = np.max(log_w)
        exp_w = np.exp(log_w - max_log_w)
        w = exp_w / np.sum(exp_w)
        P = -res.fun
        if P > P_cd:
            return s, w, P
        else:
            return s_cd, w_cd, P_cd
    except Exception:
        return s_cd, w_cd, P_cd

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Construct a discrete measure mu on [0, inf) maximizing P[X1+X2+X3 < 2*X4].
    Returns (support, weights), 1-D float arrays of equal length n >= 1.
    """
    start_time = time.time()
    deadline = start_time + 600.0 - 30.0  # 30s safety margin

    # 1. Exact optimal 2-point baseline: {0, a} with P(X=0)=q=1/4.
    #    P = 1 - q*(1-q)^3, maximized at q=1/4 -> 229/256 ~= 0.89453125.
    #    The value is independent of a (a>0); use a tiny a to stay safely positive.
    q_opt = 0.25
    a_opt = 1e-3
    s_base = np.array([0.0, a_opt], dtype=np.float64)
    w_base = np.array([q_opt, 1.0 - q_opt], dtype=np.float64)
    P_base = _compute_P_fast(s_base, w_base)

    best_P = P_base
    best_support = s_base.copy()
    best_weights = w_base.copy()

    # 2. Exact 2-point verification over a fine grid (cheap, robust).
    for q in np.linspace(0.01, 0.99, 99):
        s2 = np.array([0.0, a_opt], dtype=np.float64)
        w2 = np.array([q, 1.0 - q], dtype=np.float64)
        P2 = _compute_P_fast(s2, w2)
        if P2 > best_P:
            best_P = P2
            best_support = s2.copy()
            best_weights = w2.copy()

    # 3. 3-point structured starting points based on known good patterns
    #    Pattern: {0, 1, M} with weights tuned
    three_point_starts = []
    for M in [2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]:
        for w0 in [0.2, 0.25, 0.3, 0.35, 0.4]:
            for w1 in [0.1, 0.15, 0.2, 0.25, 0.3]:
                w2 = 1.0 - w0 - w1
                if w2 > 0.01:
                    s3 = np.array([0.0, 1.0, M], dtype=np.float64)
                    w3 = np.array([w0, w1, w2], dtype=np.float64)
                    three_point_starts.append((s3, w3))
    
    for s3, w3 in three_point_starts:
        if time.time() > deadline - 300.0:
            break
        P3 = _compute_P_fast(s3, w3)
        if P3 > best_P:
            best_P = P3
            best_support = s3.copy()
            best_weights = w3.copy()

    # 4. Time-boxed exploration of larger supports (only overwrites if it beats baseline).
    supports_to_try = []
    
    # 3-point structures with various spacings
    for a in [0.5, 1.0, 2.0, 5.0]:
        for b in [1.0, 2.0, 5.0, 10.0, 20.0]:
            s = np.array([0.0, a, b], dtype=np.float64)
            supports_to_try.append(s)
    
    # 4-point structures
    for a in [0.5, 1.0, 2.0]:
        for b in [2.0, 5.0, 10.0]:
            for c in [5.0, 10.0, 20.0, 50.0]:
                if c > b > a > 0:
                    s = np.array([0.0, a, b, c], dtype=np.float64)
                    supports_to_try.append(s)
    
    # 5-point structures
    for a in [0.5, 1.0]:
        for b in [2.0, 5.0]:
            for c in [5.0, 10.0]:
                for d in [10.0, 20.0, 50.0]:
                    if d > c > b > a > 0:
                        s = np.array([0.0, a, b, c, d], dtype=np.float64)
                        supports_to_try.append(s)

    # Geometric progressions
    for n in [3, 4, 5, 6]:
        for base in [2.0, 3.0, 5.0, 10.0]:
            s = np.array([base ** i for i in range(n)], dtype=np.float64)
            supports_to_try.append(s)
            
    # Arithmetic progressions
    for n in [3, 4, 5]:
        for step in [1.0, 2.0, 5.0]:
            s = np.arange(n, dtype=np.float64) * step
            supports_to_try.append(s)
            
    # Start with 0, then exponential
    for n in [3, 4, 5]:
        s = np.array([0.0] + [2.0 ** i for i in range(1, n)], dtype=np.float64)
        supports_to_try.append(s)
        
    # Clusters near 0 and one far point (mimicking 2-point optimal)
    for n in [3, 4, 5]:
        s = np.zeros(n)
        s[0] = 0.0
        s[1] = 1e-3
        s[2] = 1e-2
        if n > 3:
            s[3] = 1.0
        if n > 4:
            s[4] = 10.0
        supports_to_try.append(s)

    candidates = []
    for s in supports_to_try:
        if time.time() > deadline - 180.0:
            break
        n_s = len(s)
        # Initial weights: multiple strategies
        initial_ws = [
            np.ones(n_s) / n_s,
            np.array([0.25] + [0.75/(n_s-1)]*(n_s-1)) if n_s > 1 else np.ones(n_s),
        ]
        # For 3+ points, also try weight patterns biased toward small values
        if n_s >= 3:
            initial_ws.append(np.array([0.3, 0.2, 0.5/(n_s-2)] + [0.5/(n_s-2)]*(n_s-3)) if n_s > 3 else np.array([0.3, 0.2, 0.5]))
        
        for w0 in initial_ws:
            if time.time() > deadline - 150.0:
                break
            if len(w0) != n_s:
                continue
            s_opt, w_opt, P_opt = _optimize_weights_l_bfgs(s, w0, maxiter=300)
            if s_opt is not None and P_opt > best_P:
                best_P = P_opt
                best_support = s_opt.copy()
                best_weights = w_opt.copy()
                candidates.append((P_opt, s_opt.copy(), w_opt.copy()))

    # 5. Joint optimization on top candidates
    if time.time() < deadline - 90.0:
        candidates.sort(key=lambda x: -x[0])
        top_candidates = candidates[:5] if len(candidates) >= 5 else candidates
        for P_c, s_c, w_c in top_candidates:
            if time.time() > deadline - 40.0:
                break
            s_j, w_j, P_j = _joint_optimize_l_bfgs(s_c, w_c, maxiter=500)
            if s_j is not None and P_j > best_P:
                best_P = P_j
                best_support = s_j.copy()
                best_weights = w_j.copy()

    # 6. Random restarts
    if time.time() < deadline - 30.0:
        np.random.seed(42)
        for _ in range(10):
            if time.time() > deadline - 15.0:
                break
            n = np.random.randint(3, 8)
            scale = np.random.uniform(1.0, 20.0)
            s_raw = np.sort(np.random.uniform(0, scale, n))
            for i in range(1, n):
                if s_raw[i] - s_raw[i - 1] < 0.01:
                    s_raw[i] = s_raw[i - 1] + 0.01
            s_opt, w_opt, P_opt = _optimize_weights_l_bfgs(s_raw, np.ones(n)/n, maxiter=300)
            if s_opt is not None and P_opt > best_P:
                best_P = P_opt
                best_support = s_opt.copy()
                best_weights = w_opt.copy()

    # 7. Final high-precision polish pass
    if time.time() < deadline - 10.0:
        # Polish weights for the best support found
        w_polished, P_polished = _polish_weights(best_support, best_weights, maxiter=3000)
        if P_polished > best_P:
            best_P = P_polished
            best_weights = w_polished

        # If we have time, do a joint polish
        if time.time() < deadline - 5.0 and len(best_support) > 2:
            s_polished, w_polished, P_polished = _polish_joint(best_support, best_weights, maxiter=3000)
            if s_polished is not None and P_polished > best_P:
                best_P = P_polished
                best_support = s_polished
                best_weights = w_polished

    # Ensure validity
    if len(best_support) != len(best_weights):
        best_support = np.array([0.0, a_opt])
        best_weights = np.array([q_opt, 1.0 - q_opt])
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
