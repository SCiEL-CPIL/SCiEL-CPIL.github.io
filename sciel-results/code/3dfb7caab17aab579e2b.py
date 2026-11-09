# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution


def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] for discrete measure efficiently."""
    n = len(support)
    if n == 0:
        return 0.0
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    
    # Build pairwise sums s_i + s_j
    s_col = support.reshape(-1, 1)
    s_row = support.reshape(1, -1)
    pair_sums = s_col + s_row
    pair_weights = np.outer(w, w)
    
    # For each k, compute sum_{i,j} w_i w_j I(s_i + s_j < 2*s_k)
    # Then P = sum_k w_k * (above)
    P = 0.0
    for k in range(n):
        threshold = 2.0 * support[k]
        mask = pair_sums < threshold
        P += w[k] * np.sum(pair_weights * mask)
    return P


def _compute_P_fast(support, weights):
    """Faster computation using sorted approach for larger n."""
    n = len(support)
    if n == 0:
        return 0.0
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    
    # Sort support and weights together
    order = np.argsort(support)
    s_sorted = support[order]
    w_sorted = w[order]
    
    # Build pairwise sums in sorted order
    s_col = s_sorted.reshape(-1, 1)
    s_row = s_sorted.reshape(1, -1)
    pair_sums = s_col + s_row
    pair_weights = np.outer(w_sorted, w_sorted)
    
    # For each k (in sorted order), find pairs with sum < 2*s_k
    P = 0.0
    for k in range(n):
        threshold = 2.0 * s_sorted[k]
        mask = pair_sums < threshold
        P += w_sorted[k] * np.sum(pair_weights * mask)
    return P


def _neg_P(x, n):
    """Objective function for optimization: negative P."""
    s = x[:n]
    log_w = x[n:]
    
    # Ensure non-negative support
    s = np.maximum(s, 0.0)
    
    # Softplus for weights to ensure positivity
    w = np.exp(np.logaddexp(0.0, log_w))
    
    P_val = _compute_P(s, w)
    return -P_val


def _decode(x, n):
    """Decode parameters to support and weights."""
    s = x[:n]
    log_w = x[n:]
    
    s = np.maximum(s, 0.0)
    w = np.exp(np.logaddexp(0.0, log_w))
    
    return s, w


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill
    
    best_s = None
    best_w = None
    best_P = -1.0
    
    def update_best(s, w):
        nonlocal best_s, best_w, best_P
        s = np.maximum(s, 0.0)
        w = np.maximum(w, 0.0)
        if np.sum(w) <= 1e-9:
            return
        P_val = _compute_P(s, w)
        if P_val > best_P:
            best_P = P_val
            best_s = s.copy()
            best_w = w.copy()
    
    # Try multiple n values with differential evolution
    n_values = [4, 5, 6, 7, 8, 10]
    
    for n in n_values:
        if time.time() > deadline - 60.0:
            break
        
        # Define bounds for differential evolution
        # Support: [0, 100] (can be larger but let's start reasonable)
        # Log weights: [-20, 5] (so weights range from ~2e-9 to ~148)
        bounds = [(0.0, 100.0)] * n + [(-20.0, 5.0)] * n
        
        # Initial population strategy: use geometric spacing
        def init_pop(N, d):
            pop = np.zeros((N, 2 * n))
            for i in range(N):
                # Random geometric-ish spacing for support
                ratio = np.random.uniform(1.5, 4.0)
                base = np.random.uniform(0.1, 2.0)
                s_init = base * (ratio ** np.arange(n))
                s_init = np.minimum(s_init, 100.0)
                s_init = np.maximum(s_init, 0.0)
                pop[i, :n] = s_init
                # Random log weights
                pop[i, n:] = np.random.uniform(-10.0, 2.0, size=n)
            return pop
        
        try:
            pop = init_pop(30, 2 * n)
            
            # Time-limited differential evolution
            max_iter = 200
            tol = 1e-6
            
            def callback(xk, convergence):
                nonlocal best_P
                s, w = _decode(xk, n)
                P_val = _compute_P(s, w)
                if P_val > best_P:
                    update_best(s, w)
                return time.time() > deadline - 30.0
            
            result = differential_evolution(
                _neg_P, bounds, args=(n,),
                maxiter=max_iter, tol=tol,
                popsize=30, init=pop,
                callback=callback,
                seed=42
            )
            
            if result.success or result.x is not None:
                s, w = _decode(result.x, n)
                update_best(s, w)
                
        except Exception:
            pass
    
    # If we haven't found anything good, try a few more targeted searches
    if best_P < 0.5:
        # Try specific configurations known to work well
        configs = [
            # Geometric with large ratio
            ([0.0, 1.0, 10.0, 100.0], [0.5, 0.3, 0.15, 0.05]),
            # More spread
            ([0.0, 0.5, 5.0, 50.0], [0.4, 0.3, 0.2, 0.1]),
            # Three points
            ([0.0, 1.0, 100.0], [0.6, 0.3, 0.1]),
            # Five points geometric
            ([0.0, 0.5, 2.5, 12.5, 62.5], [0.4, 0.25, 0.2, 0.1, 0.05]),
        ]
        
        for s_cfg, w_cfg in configs:
            s_arr = np.array(s_cfg, dtype=float)
            w_arr = np.array(w_cfg, dtype=float)
            update_best(s_arr, w_arr)
    
    # Final local refinement around best solution
    if best_s is not None and time.time() < deadline - 30.0:
        n_best = len(best_s)
        # Normalize weights for log-space
        w_norm = best_w / np.sum(best_w)
        log_w_init = np.log(np.maximum(w_norm, 1e-10))
        x_init = np.concatenate([best_s, log_w_init])
        
        try:
            from scipy.optimize import minimize
            
            def _neg_P_fixed_n(x):
                s = np.maximum(x[:n_best], 0.0)
                w = np.exp(np.logaddexp(0.0, x[n_best:]))
                return -_compute_P(s, w)
            
            result = minimize(
                _neg_P_fixed_n, x_init, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.x is not None:
                s, w = _decode(result.x, n_best)
                update_best(s, w)
                
        except Exception:
            pass
    
    # Ensure valid output
    if best_s is None or best_w is None:
        # Fallback: simple valid measure
        best_s = np.array([0.0, 1.0, 10.0])
        best_w = np.array([0.5, 0.3, 0.2])
    
    best_s = np.maximum(best_s, 0.0)
    best_w = np.maximum(best_w, 0.0)
    
    if np.sum(best_w) <= 1e-9:
        best_s = np.array([0.0])
        best_w = np.array([1.0])
    
    return best_s, best_w


# EVOLVE-BLOCK-END
