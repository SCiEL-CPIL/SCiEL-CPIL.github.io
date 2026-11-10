# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize

def _entropy_of_projection(P, r):
    """Shannon entropy (bits) of the projected distribution x + r*y."""
    n = P.shape[0]
    xs = np.arange(n, dtype=np.float64)
    ys = np.arange(n, dtype=np.float64)
    vals = xs[None, :] + r * ys[:, None]
    uniq, inv = np.unique(vals, return_inverse=True)
    pooled = np.bincount(inv.ravel(), weights=P.ravel(), minlength=len(uniq))
    pooled = pooled[pooled > 0.0]
    return float(-(pooled * np.log2(pooled)).sum())

def _ratio(P):
    D = _entropy_of_projection(P, -1.0)
    M = max(_entropy_of_projection(P, 0.0),
            _entropy_of_projection(P, 1.0),
            _entropy_of_projection(P, 2.0))
    if M <= 1e-9:
        return -np.inf
    return D / M

def _objective(log_w):
    """Objective function for optimization. Minimizes negative ratio."""
    w = log_w - np.max(log_w)
    p = np.exp(w)
    s = p.sum()
    if s < 1e-300:
        return 1e10
    P = p / s
    P = P.reshape(10, 10)
    r = _ratio(P)
    if not np.isfinite(r):
        return 1e10
    return -r

def _polish(P_init):
    """Local refinement using L-BFGS-B."""
    log_w_init = np.log(np.clip(P_init.ravel(), 1e-300, None))
    try:
        res = minimize(_objective, log_w_init, method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-14, 'maxls': 50})
        log_w_opt = res.x
        w = log_w_opt - np.max(log_w_opt)
        p = np.exp(w)
        P_opt = p / p.sum()
        P_opt = P_opt.reshape(10, 10)
        return P_opt
    except Exception:
        return P_init

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    # Bounds for log-weights. 
    # We allow a wide range of relative weights.
    # Since the objective is invariant to scaling, we fix the max log_w to 0 
    # implicitly in the objective, but DE needs explicit bounds.
    # We can bound log_w between -10 and 0 (since max is subtracted, effectively -10..0 range of relative log weights)
    # Actually, it's safer to bound the raw log weights and let the objective handle normalization.
    # Let's use bounds for the 100 variables.
    # A uniform distribution has log_w = 0 for all.
    # Let's allow log_w in [-15, 15].
    bounds = [(-15.0, 15.0)] * (n * n)
    
    # Initial population size and iterations
    # DE is stochastic, so we set a seed for reproducibility in evaluation context if possible, 
    # but the task implies a single call. We use a fixed seed for consistency.
    np.random.seed(42)
    
    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            seed=42,
            maxiter=300,
            popsize=15,
            tol=1e-10,
            polish=False, # We will polish manually
            init='sobol'
        )
        
        log_w_best = res.x
        w = log_w_best - np.max(log_w_best)
        p = np.exp(w)
        P_best = p / p.sum()
        P_best = P_best.reshape(n, n)
        
        # Polish the result
        P_polished = _polish(P_best)
        
        # Verify and finalize
        P_final = np.clip(P_polished, 0.0, None)
        s = P_final.sum()
        if s <= 1e-9:
            P_final = np.ones((n, n)) / (n * n)
        else:
            P_final = P_final / s
            
        return P_final
        
    except Exception:
        # Fallback to a simple heuristic if optimization fails
        P = np.ones((n, n)) / (n * n)
        return P
# EVOLVE-BLOCK-END
