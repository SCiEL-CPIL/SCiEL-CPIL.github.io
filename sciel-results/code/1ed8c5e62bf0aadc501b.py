# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

_START = time.time()
_DEADLINE = 600.0


def _deadline_ok(margin=30.0):
    return (time.time() - _START) < (_DEADLINE - margin)


def _compute_P_sorted(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] using sorted pair-sums with cumulative weights.
    O(n^2 log n) per call instead of O(n^4).
    
    P = sum_{k,l} p_k * p_l * Q(2*s_l - s_k)
    where Q(t) = sum_{i,j} p_i * p_j * I(s_i + s_j < t)
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    w_sum = w.sum()
    if w_sum <= 1e-12:
        return 0.0
    p = w / w_sum
    
    # Build all pair sums and their weights
    pair_sums = s[:, None] + s[None, :]  # (n, n)
    pair_weights = p[:, None] * p[None, :]  # (n, n)
    
    # Flatten and sort by pair sum
    ps_flat = pair_sums.ravel()
    pw_flat = pair_weights.ravel()
    
    sort_idx = np.argsort(ps_flat, kind='mergesort')
    ps_sorted = ps_flat[sort_idx]
    pw_sorted = pw_flat[sort_idx]
    
    # Cumulative sum of weights
    cw_sorted = np.cumsum(pw_sorted)
    
    # For each (k, l), threshold T = 2*s_l - s_k
    # Q(T) = sum of pw_sorted where ps_sorted < T
    # Use searchsorted to find the index
    T = 2.0 * s[None, :] - s[:, None]  # (k, l), shape (n, n)
    T_flat = T.ravel()
    
    # searchsorted gives index where T_flat would be inserted to maintain order
    # We want count of ps_sorted < T_flat, so use 'left'
    idx = np.searchsorted(ps_sorted, T_flat, side='left')
    # Q values: if idx == 0, Q = 0; else Q = cw_sorted[idx-1]
    Q_vals = np.zeros(len(idx))
    mask = idx > 0
    Q_vals[mask] = cw_sorted[idx[mask] - 1]
    
    Q_matrix = Q_vals.reshape(n, n)
    
    # P = sum_{k,l} p_k * p_l * Q[k,l]
    P = np.sum(p[:, None] * p[None, :] * Q_matrix)
    return float(P)


def _project_simplex(v):
    """Project vector v onto the probability simplex."""
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    # Use the standard algorithm
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    w_sum = w.sum()
    if w_sum <= 1e-15:
        return np.ones(n) / n
    return w / w_sum


def _neg_P_from_params(params, n):
    """
    Negative P from parameter vector.
    params[0:n] = log support points (sorted)
    params[n:2n] = unnormalized weights
    """
    log_s = params[:n]
    w_raw = params[n:2*n]
    
    # Ensure support is sorted and positive
    s = np.exp(log_s)
    # Sort support and corresponding weights
    order = np.argsort(s)
    s = s[order]
    w_raw = w_raw[order]
    
    # Project weights to simplex
    w = _project_simplex(w_raw)
    
    # Check validity
    if np.any(s < -1e-9) or np.any(np.isnan(s)) or np.any(np.isinf(s)):
        return 1.0
    if np.any(w < -1e-9) or np.any(np.isnan(w)) or np.any(np.isinf(w)):
        return 1.0
    
    return -_compute_P_sorted(s, w)


def _optimize_n_points(n, log_min, log_max, max_iter, pop_size, tol):
    """
    Use differential evolution to optimize support and weights for n points.
    """
    # Bounds for log support: [log_min, log_max]
    # To ensure sorted, we'll work with differences or just let DE handle it
    # Actually, let's use log support directly and sort after
    
    # Initial bounds
    lb = np.array([log_min] * n + [0.0] * n)
    ub = np.array([log_max] * n + [10.0] * n)
    
    # For weights, use log-space to ensure positivity
    # w_i = exp(w_raw_i) / sum(exp(w_raw_j))
    # But DE works better with bounded params. Let's use a different approach:
    # Use Dirichlet-like parameterization via softmax
    
    # Actually, let's just use raw weights and project to simplex
    # DE will explore the space
    
    bounds = list(zip(lb, ub))
    
    # Callback to check deadline
    def callback(xk, convergence=True):
        return not _deadline_ok(margin=90.0)
    
    try:
        result = differential_evolution(
            _neg_P_from_params,
            bounds,
            args=(n,),
            maxiter=max_iter,
            popsize=pop_size,
            tol=tol,
            mutation=(0.3, 0.9),
            recombination=0.7,
            seed=42,
            callback=callback,
            polish=False,  # Disable polishing to save time; we'll do it separately
            init='latinhypercube'
        )
        
        if result.success or result.fun < 1.0:
            params = result.x
            log_s = params[:n]
            w_raw = params[n:2*n]
            s = np.exp(log_s)
            order = np.argsort(s)
            s = s[order]
            w_raw = w_raw[order]
            w = _project_simplex(w_raw)
            val = _compute_P_sorted(s, w)
            return val, s, w
        else:
            return 0.0, None, None
    except Exception:
        return 0.0, None, None


def _refine_with_scipy(support, weights, n):
    """
    Refine a solution using scipy.optimize.minimize (L-BFGS-B) on the parameter space.
    """
    def params_to_sw(params):
        log_s = params[:n]
        w_raw = params[n:2*n]
        s = np.exp(log_s)
        order = np.argsort(s)
        s = s[order]
        w_raw = w_raw[order]
        w = _project_simplex(w_raw)
        return s, w
    
    def objective(params):
        s, w = params_to_sw(params)
        return -_compute_P_sorted(s, w)
    
    # Initial params
    log_s0 = np.log(support)
    # Use log-weights as initial (they'll be projected)
    w0 = weights + 1e-10
    log_w0 = np.log(w0)
    
    x0 = np.concatenate([log_s0, log_w0])
    
    bounds = [(np.log(1e-6), np.log(1e4))] * n + [(-10.0, 10.0)] * n
    
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if result.fun < 1.0:
            s, w = params_to_sw(result.x)
            val = _compute_P_sorted(s, w)
            return val, s, w
        else:
            return 0.0, None, None
    except Exception:
        return 0.0, None, None


def _try_3point_family():
    """
    Try the 3-point family {a, b, c} with optimized parameters using DE.
    This is a structured family that has been effective.
    """
    def neg_P_3pt(params):
        a, b, c = params[0], params[1], params[2]
        # Ensure a < b < c and all positive
        if a <= 0 or b <= 0 or c <= 0:
            return 1.0
        if not (a < b < c):
            return 1.0
        
        # Weights: use 3 parameters on simplex
        w_raw = params[3:6]
        w = _project_simplex(w_raw)
        
        s = np.array([a, b, c])
        return -_compute_P_sorted(s, w)
    
    bounds = [(1e-4, 10.0), (1e-2, 100.0), (1.0, 1000.0),
              (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    
    try:
        result = differential_evolution(
            neg_P_3pt,
            bounds,
            maxiter=100,
            popsize=20,
            tol=1e-8,
            seed=123,
            polish=False,
            callback=lambda xk, convergence=True: not _deadline_ok(margin=90.0),
            init='latinhypercube'
        )
        
        if result.fun < 1.0:
            a, b, c = result.x[0], result.x[1], result.x[2]
            w_raw = result.x[3:6]
            w = _project_simplex(w_raw)
            s = np.array([a, b, c])
            val = _compute_P_sorted(s, w)
            return val, s, w
        else:
            return 0.0, None, None
    except Exception:
        return 0.0, None, None


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    # Structural change: Reformulate as uniform distribution on [0, L]
    # Using a discrete approximation with n points
    n = 100
    L = 1.0
    support = np.linspace(0.0, L, n, dtype=np.float64)
    weights = np.ones(n, dtype=np.float64)
    
    return support, weights
# EVOLVE-BLOCK-END
