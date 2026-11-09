# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import sys


def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] efficiently.
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return 0.0
    w = w / w_sum
    
    # Sort by support
    order = np.argsort(s)
    s = s[order]
    w = w[order]
    
    # Pair sums and probabilities
    pair_sums = s[:, None] + s[None, :]  # (n, n)
    pair_probs = np.outer(w, w)  # (n, n)
    
    # Thresholds: threshold[k, l] = 2*s_l - s_k
    thresholds = 2.0 * s[None, :] - s[:, None]  # (n, n)
    
    # For each (k, l), compute inner_sum = sum_{i,j} pair_probs[i,j] * I(pair_sums[i,j] < thresholds[k,l])
    # Using broadcasting
    ps = pair_sums[None, None, :, :]  # (1, 1, n, n)
    th = thresholds[:, :, None, None]  # (n, n, 1, 1)
    indicator = (ps < th).astype(np.float64)  # (n, n, n, n)
    
    pp = pair_probs[None, None, :, :]  # (1, 1, n, n)
    inner_sum = np.sum(pp * indicator, axis=(2, 3))  # (n, n)
    
    wl = np.outer(w, w)
    P = np.sum(wl * inner_sum)
    
    return P


def _neg_P_from_params(params, n):
    """
    Objective: minimize -P.
    params: [s_0, ..., s_{n-1}, log_w_0, ..., log_w_{n-1}]
    Weights are derived from log-weights to ensure positivity.
    """
    s = params[:n]
    log_w = params[n:]
    
    # Ensure support is non-negative
    s = np.maximum(s, 0.0)
    
    # Convert log-weights to weights, normalize
    w = np.exp(log_w)
    w_sum = np.sum(w)
    if w_sum <= 1e-15:
        return 1.0  # Invalid
    w = w / w_sum
    
    return -_compute_P_fast(s, w)


def _neg_P_from_params_clipped(params, n, s_max=50.0):
    """
    Objective with clipped support.
    """
    s = params[:n]
    log_w = params[n:]
    
    s = np.clip(s, 0.0, s_max)
    w = np.exp(log_w)
    w_sum = np.sum(w)
    if w_sum <= 1e-15:
        return 1.0
    w = w / w_sum
    
    return -_compute_P_fast(s, w)


def _optimize_n(n, start_time, deadline, s_max=50.0):
    """
    Optimize for a given n using differential evolution + Nelder-Mead.
    Returns (support, weights, P) or (None, None, 0.0).
    """
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining < 5:
        return None, None, 0.0
    
    # Bounds: support in [0, s_max], log-weights in [-20, 5]
    bounds_s = [(0.0, s_max)] * n
    bounds_logw = [(-20.0, 5.0)] * n
    bounds = bounds_s + bounds_logw
    
    # Initial guess: spread out support, equal log-weights
    x0 = np.concatenate([
        np.linspace(0.5, s_max * 0.5, n),
        np.zeros(n)
    ])
    
    try:
        # Differential evolution
        result_de = differential_evolution(
            _neg_P_from_params_clipped, bounds, args=(n, s_max),
            maxiter=min(100, int(remaining / 5)), 
            popsize=min(15, max(5, n + 5)),
            tol=1e-8,
            seed=42,
            polish=True,
            init='sobol'
        )
        
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining < 10:
            best_params = result_de.x
        else:
            # Polish with Nelder-Mead
            result_nm = minimize(
                _neg_P_from_params_clipped, result_de.x, args=(n, s_max),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            best_params = result_nm.x
        
        s = np.clip(best_params[:n], 0.0, s_max)
        log_w = best_params[n:]
        w = np.exp(log_w)
        w_sum = np.sum(w)
        if w_sum <= 1e-15:
            return None, None, 0.0
        w = w / w_sum
        
        P = _compute_P_fast(s, w)
        return s, w, P
        
    except Exception:
        return None, None, 0.0


def _try_geometric_structure(n, start_time, deadline):
    """
    Try a geometric structure: support at {0, r, r^2, ..., r^{n-1}} or similar.
    Optimize r and weights.
    """
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining < 5:
        return None, None, 0.0
    
    def _neg_P_geo(params, n):
        r = params[0]
        log_w = params[1:]
        r = max(r, 0.01)
        r = min(r, 10.0)
        s = r ** np.arange(n, dtype=np.float64)
        w = np.exp(log_w)
        w_sum = np.sum(w)
        if w_sum <= 1e-15:
            return 1.0
        w = w / w_sum
        return -_compute_P_fast(s, w)
    
    bounds = [(0.01, 10.0)] + [(-20.0, 5.0)] * n
    
    x0 = np.concatenate([[1.5], np.zeros(n)])
    
    try:
        result = differential_evolution(
            _neg_P_geo, bounds, args=(n,),
            maxiter=min(50, int(remaining / 5)),
            popsize=min(10, n + 5),
            seed=42,
            polish=True
        )
        
        r = np.clip(result.x[0], 0.01, 10.0)
        log_w = result.x[1:]
        s = r ** np.arange(n, dtype=np.float64)
        w = np.exp(log_w)
        w_sum = np.sum(w)
        if w_sum <= 1e-15:
            return None, None, 0.0
        w = w / w_sum
        
        P = _compute_P_fast(s, w)
        return s, w, P
        
    except Exception:
        return None, None, 0.0


def _try_bimodal_structure(n, start_time, deadline):
    """
    Try a bimodal structure: half mass near 0, half mass at some value a.
    Optimize a and the split.
    """
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining < 5:
        return None, None, 0.0
    
    # Support: {0, eps, 2*eps, ..., (n/2)*eps, a - eps, a - 2*eps, ..., a}
    # Or simpler: {0, a} with n points spread
    def _neg_P_bi(params, n):
        a = params[0]
        log_w = params[1:]
        a = max(a, 0.01)
        a = min(a, 100.0)
        # Half near 0, half near a
        half = n // 2
        s_first = np.linspace(0, 0.1, half) if half > 0 else np.array([0.0])
        s_second = np.linspace(a - 0.1, a, n - half) if n - half > 0 else np.array([a])
        s = np.concatenate([s_first, s_second])
        w = np.exp(log_w)
        w_sum = np.sum(w)
        if w_sum <= 1e-15:
            return 1.0
        w = w / w_sum
        return -_compute_P_fast(s, w)
    
    bounds = [(0.01, 100.0)] + [(-20.0, 5.0)] * n
    x0 = np.concatenate([[5.0], np.zeros(n)])
    
    try:
        result = differential_evolution(
            _neg_P_bi, bounds, args=(n,),
            maxiter=min(50, int(remaining / 5)),
            popsize=min(10, n + 5),
            seed=42,
            polish=True
        )
        
        a = np.clip(result.x[0], 0.01, 100.0)
        log_w = result.x[1:]
        half = n // 2
        s_first = np.linspace(0, 0.1, half) if half > 0 else np.array([0.0])
        s_second = np.linspace(a - 0.1, a, n - half) if n - half > 0 else np.array([a])
        s = np.concatenate([s_first, s_second])
        w = np.exp(log_w)
        w_sum = np.sum(w)
        if w_sum <= 1e-15:
            return None, None, 0.0
        w = w / w_sum
        
        P = _compute_P_fast(s, w)
        return s, w, P
        
    except Exception:
        return None, None, 0.0


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = 620.0  # Safety margin before 630s limit
    
    best_support = None
    best_weights = None
    best_P = 0.0
    
    # Baseline: uniform on {0, 1, ..., 15}
    n_base = 16
    support_base = np.arange(n_base, dtype=np.float64)
    weights_base = np.ones(n_base, dtype=np.float64) / n_base
    P_base = _compute_P_fast(support_base, weights_base)
    best_support = support_base.copy()
    best_weights = weights_base.copy()
    best_P = P_base
    
    # Try various n values with full optimization
    for n in [3, 4, 5, 6, 7, 8]:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        
        s, w, P = _optimize_n(n, start_time, deadline, s_max=50.0)
        if s is not None and P > best_P:
            best_support = s
            best_weights = w
            best_P = P
    
    # Try geometric structure
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        for n in [4, 5, 6]:
            s, w, P = _try_geometric_structure(n, start_time, deadline)
            if s is not None and P > best_P:
                best_support = s
                best_weights = w
                best_P = P
    
    # Try bimodal structure
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        for n in [4, 6, 8]:
            s, w, P = _try_bimodal_structure(n, start_time, deadline)
            if s is not None and P > best_P:
                best_support = s
                best_weights = w
                best_P = P
    
    # Final validation
    if best_support is None or best_weights is None:
        return support_base, weights_base
    
    # Ensure no NaN or inf
    if not np.all(np.isfinite(best_support)) or not np.all(np.isfinite(best_weights)):
        return support_base, weights_base
    
    # Ensure support >= 0
    best_support = np.maximum(best_support, 0.0)
    # Ensure weights >= 0
    best_weights = np.maximum(best_weights, 1e-12)
    
    # Ensure sum of weights > 1e-9
    if np.sum(best_weights) <= 1e-9:
        return support_base, weights_base
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
