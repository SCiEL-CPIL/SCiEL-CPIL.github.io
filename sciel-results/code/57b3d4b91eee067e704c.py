"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

_START_TIME = time.time()
_TIME_LIMIT = 620.0  # safety margin before 630s kill

# Precompute Gauss-Laguerre quadrature nodes and weights (alpha=1, weight x*exp(-x))
# For degree 24 polynomial (12 double roots), we need sufficient nodes for accuracy
_N_QUAD = 80
try:
    from scipy.special import roots_genlaguerre
    _QUAD_NODES, _QUAD_WEIGHTS = roots_genlaguerre(_N_QUAD, 1)
    _QUAD_LOG_WEIGHTS = np.log(_QUAD_WEIGHTS + 1e-300)
except Exception:
    from numpy.polynomial.laguerre import laggauss
    _QUAD_NODES, _QUAD_WEIGHTS_STD = laggauss(_N_QUAD)
    _QUAD_WEIGHTS = _QUAD_WEIGHTS_STD * _QUAD_NODES
    _QUAD_LOG_WEIGHTS = np.log(_QUAD_WEIGHTS + 1e-300)

def _objective_fast(r):
    """
    Fast objective using precomputed generalized Gauss-Laguerre quadrature.
    Computes ∫_0^∞ |P(x)|^2 * x * e^{-x} dx where P(x) = ∏(x - r_i)^2.
    """
    r = np.asarray(r, dtype=np.float64)
    if np.any(r <= 0) or np.any(np.isnan(r)):
        return 1e12
    
    nodes = _QUAD_NODES
    log_weights = _QUAD_LOG_WEIGHTS
    
    # Compute log|P(x)|^2 = 2 * sum_i log|x - r_i|^2 = 2 * sum_i 2*log|x - r_i|
    # = 4 * sum_i log|x - r_i|
    # Actually P(x) = prod (x-ri)^2, so |P|^2 = prod (x-ri)^4
    # log|P|^2 = 4 * sum_i log|x - r_i|
    
    log_abs_P_sq = np.zeros_like(nodes)
    for ri in r:
        diff = nodes - ri
        # Avoid log(0) by adding tiny epsilon
        log_abs_P_sq += 4.0 * np.log(np.abs(diff) + 1e-300)
    
    val = log_abs_P_sq + log_weights
    if np.any(np.isnan(val)) or np.any(np.isinf(val)):
        return 1e12
    
    max_val = np.max(val)
    if max_val > 700:  # overflow risk
        return 1e12
    integral = np.sum(np.exp(val - max_val)) * np.exp(max_val)
    
    return integral

def _run_optimization(initial_r, max_iter=5000, ftol=1e-12, gtol=1e-10):
    """Run scipy optimization with time guard."""
    from scipy.optimize import minimize
    
    def obj(x):
        if time.time() - _START_TIME > _TIME_LIMIT:
            raise TimeoutError()
        return _objective_fast(x)
    
    try:
        result = minimize(
            obj,
            initial_r,
            method='L-BFGS-B',
            bounds=[(0.01, None)] * 12,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        return result.x, result.fun
    except TimeoutError:
        return initial_r, _objective_fast(initial_r)
    except Exception:
        return initial_r, _objective_fast(initial_r)

def _polish(r, max_iter=2000):
    """Final high-precision polish pass with tighter tolerances."""
    if time.time() - _START_TIME > _TIME_LIMIT - 10:
        return r, _objective_fast(r)
    
    try:
        r_opt, val = _run_optimization(
            r, max_iter=max_iter, ftol=1e-15, gtol=1e-13
        )
        return r_opt, val
    except Exception:
        return r, _objective_fast(r)

def _multiple_restarts():
    """Run multiple random restarts to avoid local minima."""
    best_r = None
    best_val = np.inf
    
    starts = []
    
    # Seed: evenly spaced
    starts.append(np.array([4.0 * (i + 1) for i in range(12)]))
    
    # Zeros of Laguerre polynomial L_12(x) (standard, alpha=0)
    from scipy.special import roots_laguerre
    try:
        zeros, _ = roots_laguerre(12)
        starts.append(zeros.copy())
    except:
        pass
    
    # Zeros of generalized Laguerre L_12^(1)(x) (alpha=1)
    from scipy.special import roots_genlaguerre
    try:
        zeros1, _ = roots_genlaguerre(12, 1)
        starts.append(zeros1.copy())
    except:
        pass
    
    # Random starts with different seeds
    for seed in [42, 123, 456, 789, 1011]:
        np.random.seed(seed)
        r = np.sort(np.random.uniform(0.5, 50.0, 12))
        starts.append(r)
    
    # Spaced starts with different spacings
    for spacing in [2.0, 3.0, 5.0]:
        r = spacing * np.arange(1, 13)
        starts.append(r)
    
    # Clustered: some close together
    starts.append(np.array([1, 2, 3, 5, 8, 13, 21, 34, 40, 45, 50, 55.0]))
    
    # More spread out
    starts.append(np.array([1, 3, 6, 10, 15, 22, 30, 40, 50, 60, 70, 80.0]))
    
    # Near the mean of the weight distribution (x=1 for alpha=1)
    starts.append(np.array([0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30.0]))
    
    for i, start in enumerate(starts):
        if time.time() - _START_TIME > _TIME_LIMIT - 30:
            break
        
        elapsed = time.time() - _START_TIME
        remaining = _TIME_LIMIT - elapsed - 30
        if remaining < 10:
            break
        
        # Scale max_iter based on remaining time
        # Each objective eval takes ~0.1ms, each L-BFGS-B iter ~1ms
        max_iter = int(min(5000, max(500, remaining * 500)))
        
        r_opt, val = _run_optimization(start, max_iter=max_iter)
        
        if val < best_val:
            best_val = val
            best_r = r_opt.copy()
    
    # Final polish on the best solution
    if best_r is not None:
        best_r, best_val = _polish(best_r)
    
    return best_r, best_val

def construct():
    """
    Returns list of 12 positive double-root locations for the Laguerre
    combination (m=12, n=1) that minimizes the C4 upper bound.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    try:
        best_r, best_val = _multiple_restarts()
        
        if best_r is None:
            # Fallback to seed
            return [4.0 * (i + 1) for i in range(12)]
        
        # Ensure all are positive
        best_r = np.maximum(best_r, 0.01)
        
        # Sort for consistency
        best_r = np.sort(best_r)
        
        return [float(x) for x in best_r]
        
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}")
        return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
