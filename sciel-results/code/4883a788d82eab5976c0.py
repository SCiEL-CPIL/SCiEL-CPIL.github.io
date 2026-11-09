# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly."""
    n = len(support)
    if n == 0:
        return 0.0
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    p = weights / w_sum
    
    pair_sums = support[:, None] + support[None, :]
    pair_probs = p[:, None] * p[None, :]
    thresholds = 2.0 * support[None, :] - support[:, None]
    
    total = 0.0
    for k in range(n):
        for l in range(n):
            thr = thresholds[k, l]
            mask = pair_sums < thr
            inner = np.sum(pair_probs * mask)
            total += p[k] * p[l] * inner
    
    return total


def _neg_P_from_params(params, n):
    """Compute -P from parameter vector: first n are supports (s[0]=0 fixed), next n are weights."""
    s = np.zeros(n, dtype=np.float64)
    s[1:] = params[:n-1]
    w = params[n-1:]
    
    # Ensure validity
    if np.any(s < -1e-9) or np.any(w < -1e-9):
        return 1e10
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return 1e10
    
    p = w / w_sum
    P = _compute_P(s, p)
    return -P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 620.0
    
    best_support = np.array([0.0, 1.0], dtype=np.float64)
    best_weights = np.array([0.5, 0.5], dtype=np.float64)
    best_P = _compute_P(best_support, best_weights)
    
    # Try various n values with Nelder-Mead optimization
    # s[0] is fixed at 0 by scale invariance
    n_values = [2, 3, 4, 5, 6]
    
    rng = np.random.default_rng(42)
    
    for n in n_values:
        if time.time() > deadline - 30:
            break
        
        # Number of free params: (n-1) supports + n weights = 2n-1
        n_params = 2 * n - 1
        
        # Multiple random restarts
        n_restarts = max(5, int(120 / n))  # more restarts for smaller n
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initial guess
            # Supports: sorted, positive (except s[0]=0)
            if n == 2:
                s_init = np.array([0.0, rng.uniform(0.5, 10.0)])
            else:
                s_rest = np.sort(rng.uniform(0.5, 20.0, size=n-1))
                s_init = np.concatenate([[0.0], s_rest])
            
            w_init = rng.dirichlet(np.ones(n))
            
            # Build param vector: [s_1, ..., s_{n-1}, w_0, ..., w_{n-1}]
            params_init = np.concatenate([s_init[1:], w_init])
            
            # Bounds: supports > 0, weights > 0
            bounds = [(1e-6, 1e6)] * (n-1) + [(1e-6, 1.0)] * n
            
            try:
                result = minimize(
                    _neg_P_from_params,
                    params_init,
                    args=(n,),
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                
                if result.fun < -1e9:  # invalid
                    continue
                
                P_val = -result.fun
                if P_val > best_P + 1e-12:
                    # Extract support and weights
                    s_opt = np.zeros(n, dtype=np.float64)
                    s_opt[1:] = result.x[:n-1]
                    w_opt = result.x[n-1:]
                    w_sum = np.sum(w_opt)
                    if w_sum > 1e-9:
                        w_opt = w_opt / w_sum
                        best_support = s_opt.copy()
                        best_weights = w_opt.copy()
                        best_P = P_val
            except Exception:
                continue
    
    # Final validation
    if best_P <= 0:
        # Fallback
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([0.5, 0.5], dtype=np.float64)
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
