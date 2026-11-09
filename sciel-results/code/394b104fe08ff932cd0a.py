# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_start_time = time.time()
_DEADLINE = 600.0  # seconds, with safety margin

def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] for discrete measure."""
    n = len(support)
    w = weights / weights.sum()
    # Build pairwise sums s_i + s_j
    ss = support[:, None] + support[None, :]  # n x n
    # For each (k, l): need sum over (i,j) with ss[i,j] < 2*s_l - s_k of w_i * w_j
    # P = sum_{k,l} w_k * w_l * (sum_{i,j: ss[i,j] < 2*s_l - s_k} w_i * w_j)
    # Vectorize: for each (k,l), threshold = 2*s_l - s_k
    thresholds = 2.0 * support[None, :] - support[:, None]  # n x n, [k,l]
    # For each (k,l), we need sum of w_i*w_j over ss[i,j] < thresholds[k,l]
    # Precompute cumulative: sort ss, but easier: for each threshold, count
    # Use broadcasting: ss is n x n, thresholds is n x n
    # mask[k,l,i,j] = ss[i,j] < thresholds[k,l] -> too big for large n
    # Instead: for each (k,l), compute sum
    # For small n, just do it directly
    P = 0.0
    for k in range(n):
        for l in range(n):
            thresh = 2.0 * support[l] - support[k]
            mask = ss < thresh
            inner = np.sum(w[:, None] * w[None, :] * mask)
            P += w[k] * w[l] * inner
    return P

def _objective(params, n):
    """Objective to minimize: -P. params = [s_0,...,s_{n-1}, w_0,...,w_{n-1}]"""
    s = params[:n]
    w = params[n:]
    # Enforce s >= 0, w >= 0
    s = np.maximum(s, 0.0)
    w = np.maximum(w, 0.0)
    if w.sum() < 1e-9:
        return 1e10
    P = _compute_P(s, w)
    return -P

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    best_support = None
    best_weights = None
    best_P = -1.0
    
    # Start with the 2-point analytical optimum
    p_opt = (15 - np.sqrt(33)) / 16.0
    s2 = np.array([0.0, 1.0])
    w2 = np.array([p_opt, 1.0 - p_opt])
    P2 = _compute_P(s2, w2)
    best_P = P2
    best_support = s2.copy()
    best_weights = w2.copy()
    
    # Try various n values with local optimization
    for n in [2, 3, 4, 5, 6, 7, 8]:
        if _time_left() < 60:
            break
        
        # Multiple random restarts
        for restart in range(20):
            if _time_left() < 30:
                break
            
            # Random initial point
            np.random.seed(42 + restart * 1000 + n)
            s0 = np.sort(np.random.exponential(1.0, size=n))
            w0 = np.random.dirichlet(np.ones(n))
            params0 = np.concatenate([s0, w0])
            
            # Use Powell (doesn't require gradients)
            try:
                res = minimize(_objective, params0, args=(n,), method='Powell',
                              options={'maxiter': 500, 'maxfev': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.success or res.fun < -best_P:
                    params = res.x
                    s = np.maximum(params[:n], 0.0)
                    w = np.maximum(params[n:], 0.0)
                    if w.sum() > 1e-9:
                        P = _compute_P(s, w)
                        if P > best_P:
                            best_P = P
                            best_support = s.copy()
                            best_weights = w.copy()
            except Exception:
                pass
    
    # Try differential evolution for small n if time permits
    for n in [3, 4, 5]:
        if _time_left() < 120:
            break
        
        try:
            bounds = [(0, 5.0)] * n + [(1e-6, 1.0)] * n
            res = differential_evolution(_objective, bounds, args=(n,),
                                          maxiter=100, popsize=15, seed=42,
                                          tol=1e-8, polish=True)
            if res.fun < -best_P:
                params = res.x
                s = np.maximum(params[:n], 0.0)
                w = np.maximum(params[n:], 0.0)
                if w.sum() > 1e-9:
                    P = _compute_P(s, w)
                    if P > best_P:
                        best_P = P
                        best_support = s.copy()
                        best_weights = w.copy()
        except Exception:
            pass
    
    # Final polish on best result
    if best_support is not None and _time_left() > 30:
        n = len(best_support)
        params0 = np.concatenate([best_support, best_weights])
        try:
            res = minimize(_objective, params0, args=(n,), method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < -best_P:
                params = res.x
                s = np.maximum(params[:n], 0.0)
                w = np.maximum(params[n:], 0.0)
                if w.sum() > 1e-9:
                    P = _compute_P(s, w)
                    if P > best_P:
                        best_P = P
                        best_support = s.copy()
                        best_weights = w.copy()
        except Exception:
            pass
    
    # Ensure validity
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    if best_weights.sum() <= 1e-9:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([p_opt, 1.0 - p_opt])
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)

# EVOLVE-BLOCK-END
