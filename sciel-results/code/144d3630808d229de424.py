"""Optimized program: gradient-based optimization of double-root locations to minimize C4 bound."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_genlaguerre

_START_TIME = time.time()
_HARD_DEADLINE = 560.0
M = 12
N = 1
NROOTS = 12

def _deadline_reached():
    return (time.time() - _START_TIME) >= _HARD_DEADLINE

def _compute_bound(r):
    """
    Compute the uncertainty bound C4 for given double-root locations.
    C4 = [integral_0^inf x^4 e^{-x} |P(x)|^2 dx] / [integral_0^inf e^{-x} |P(x)|^2 dx]
    where P(x) = prod_{i=1}^{12} (x - r_i)^2
    """
    r = np.asarray(r, dtype=float)
    
    # Use Gauss-Laguerre quadrature for integrals with e^{-x} weight
    x_nodes, x_weights = np.polynomial.laguerre.laggauss(60)
    
    # Evaluate P(x) at each node efficiently
    P_vals = np.ones_like(x_nodes)
    for ri in r:
        P_vals *= (x_nodes - ri) ** 2
    
    P2 = P_vals ** 2
    
    num = np.sum(x_weights * x_nodes**4 * P2)
    den = np.sum(x_weights * P2)
    
    if den < 1e-30:
        return 1e10
    
    return num / den

def _compute_bound_log(r):
    """Log-space computation to avoid overflow/underflow for very large/small values."""
    r = np.asarray(r, dtype=float)
    x_nodes, x_weights = np.polynomial.laguerre.laggauss(60)
    
    # Compute log|P(x)| = 2 * sum_i log|x - r_i|
    log_P_abs = np.zeros_like(x_nodes)
    for ri in r:
        diff = x_nodes - ri
        # Handle zeros: if x_node == ri, P(x) = 0, log is -inf
        # But for quadrature, we can treat this as very negative
        log_P_abs += 2.0 * np.log(np.abs(diff) + 1e-300)
    
    log_P2 = 2.0 * log_P_abs
    
    # Compute weighted sums in log space
    log_weights = np.log(x_weights + 1e-300)
    log_x4 = 4.0 * np.log(x_nodes + 1e-300)
    
    # num = sum(x_weights * x_nodes^4 * P2)
    # log_num_terms = log_weights + log_x4 + log_P2
    log_num_terms = log_weights + log_x4 + log_P2
    # den = sum(x_weights * P2)
    log_den_terms = log_weights + log_P2
    
    # Use log-sum-exp for numerical stability
    def logsumexp(a):
        a_max = np.max(a)
        if np.isneginf(a_max):
            return -np.inf
        return a_max + np.log(np.sum(np.exp(a - a_max)))
    
    log_num = logsumexp(log_num_terms)
    log_den = logsumexp(log_den_terms)
    
    if not np.isfinite(log_num) or not np.isfinite(log_den):
        return 1e10
    
    log_bound = log_num - log_den
    bound = np.exp(log_bound)
    
    if not np.isfinite(bound) or bound < 0:
        return 1e10
    
    return bound

def _objective(r):
    """Objective function for optimization: minimize C4 bound."""
    if _deadline_reached():
        return 1e10
    try:
        return _compute_bound_log(r)
    except Exception:
        return 1e10

def _construct_core():
    """
    Optimize root locations to minimize the C4 uncertainty bound.
    Start from Laguerre polynomial zeros and use gradient-based optimization.
    """
    # Initial guess: zeros of L_12^{(1)}(x)
    r0 = roots_genlaguerre(M, N)
    r0 = np.sort(r0.real)
    r0 = np.maximum(r0, 0.1)
    
    # Try multiple optimization methods and initial conditions
    best_result = r0.copy()
    best_score = _compute_bound(r0)
    
    # Method 1: L-BFGS-B with bounds
    bounds = [(0.01, 100.0)] * NROOTS
    
    if not _deadline_reached():
        try:
            res = minimize(_objective, r0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
    
    # Method 2: Try different initial spreads
    for scale in [0.5, 2.0, 3.0]:
        if _deadline_reached():
            break
        r_init = r0 * scale
        r_init = np.maximum(r_init, 0.1)
        try:
            res = minimize(_objective, r_init, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
    
    # Method 3: Nelder-Mead as fallback (no gradient needed)
    if not _deadline_reached():
        try:
            res = minimize(_objective, best_result, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
    
    # Ensure positivity and sort
    best_result = np.sort(best_result)
    best_result = np.maximum(best_result, 0.01)
    
    return best_result.tolist()

def construct():
    """Return 12 positive double-root locations for the Laguerre combination
    (m=12, n=1).  Score is the upper bound on C4 (lower is better)."""
    try:
        result = _construct_core()
    except Exception:
        # Fallback to Laguerre zeros
        try:
            r = roots_genlaguerre(M, N)
            result = np.sort(r.real).tolist()
        except Exception:
            result = [4.0 * (i + 1) for i in range(NROOTS)]
    
    # Hard safety: if anything ran too long, return the known-good seed roots.
    if (time.time() - _START_TIME) >= _HARD_DEADLINE:
        result = [4.0 * (i + 1) for i in range(NROOTS)]
    
    # Final sanity: exactly 12 positive, increasing floats.
    if not (isinstance(result, list) and len(result) == NROOTS):
        result = [4.0 * (i + 1) for i in range(NROOTS)]
    out = []
    last = 0.0
    for v in result:
        v = float(v)
        if not (v > 0.0 and math.isfinite(v)):
            v = last + 1.0
        if v <= last:
            v = last + 1e-6
        out.append(v)
        last = v
    return out
