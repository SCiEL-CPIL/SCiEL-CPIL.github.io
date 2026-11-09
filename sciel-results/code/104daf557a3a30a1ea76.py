"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_start_time = time.time()
_DEADLINE = 620.0  # Safety margin before 630s kill

def _compute_c4(roots):
    """Compute the C4 upper bound given 12 positive double-root locations.
    
    Uses log-space computation for numerical stability.
    C4 = P(0) / integral_0^inf P(x) * exp(-x) dx
    where P(x) = prod_{i=1}^{12} (x - r_i)^2
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12:
        return 1e10
    if np.any(roots <= 0):
        return 1e10
    if np.any(np.diff(roots) <= 0):
        return 1e10
    
    # P(0) = prod r_i^2, compute in log space
    log_p0 = 2.0 * np.sum(np.log(roots))
    
    # Numerically compute integral of P(x) * exp(-x) from 0 to inf
    # Use Gauss-Laguerre quadrature with more points for accuracy
    n_quad = 300
    x, w = np.polynomial.laguerre.laggauss(n_quad)
    
    # Compute log(P(x)) = 2 * sum log|x - r_i| for each quadrature point
    # P(x) = prod (x - r_i)^2
    # log|P(x)| = 2 * sum log|x - r_i|
    # But we need to handle sign: since roots are positive and x >= 0,
    # for x > max(roots), all terms positive
    # For x < min(roots), (x - r_i) < 0, so (x - r_i)^2 > 0
    # So P(x) > 0 always for real x not equal to roots
    # We can compute log(P(x)) = 2 * sum log|x - r_i|
    
    diff = x[:, None] - roots[None, :]
    # Use log of absolute value since squared
    log_p_vals = 2.0 * np.sum(np.log(np.abs(diff)), axis=1)
    
    # integral = sum w_i * P(x_i) * exp(-x_i)
    # log(integrand) = log(w_i) + log_p_vals[i] - x_i
    log_integrand = np.log(w) + log_p_vals - x
    
    # Sum in log space using log-sum-exp
    max_log = np.max(log_integrand)
    if not np.isfinite(max_log):
        return 1e10
    log_integral = max_log + np.log(np.sum(np.exp(log_integrand - max_log)))
    
    if not np.isfinite(log_integral):
        return 1e10
    
    log_c4 = log_p0 - log_integral
    if not np.isfinite(log_c4):
        return 1e10
    
    c4 = np.exp(log_c4)
    if not np.isfinite(c4) or c4 <= 0:
        return 1e10
    
    return c4

def _compute_c4_fast(roots):
    """Faster version using direct computation for smaller values."""
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12:
        return 1e10
    if np.any(roots <= 0):
        return 1e10
    if np.any(np.diff(roots) <= 0):
        return 1e10
    
    n_quad = 200
    x, w = np.polynomial.laguerre.laggauss(n_quad)
    
    diff = x[:, None] - roots[None, :]
    p_vals = np.prod(diff ** 2, axis=1)
    
    integrand = w * p_vals * np.exp(-x)
    integral = np.sum(integrand)
    
    if integral <= 0 or not np.isfinite(integral):
        return 1e10
    
    p0 = np.prod(roots ** 2)
    c4 = p0 / integral
    
    if not np.isfinite(c4) or c4 <= 0:
        return 1e10
    
    return c4

def construct():
    """Return 12 positive double-root locations minimizing C4."""
    global _start_time
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 30:
        # Not enough time, return default
        return [4.0 * (i + 1) for i in range(12)]
    
    # Better initial guess: zeros of Laguerre polynomial L_12(x)
    # The optimal roots for this type of problem are related to Laguerre zeros
    try:
        from numpy.polynomial.laguerre import laggauss
        # For m=12, the relevant zeros are those of the Laguerre polynomial
        # Use Gauss-Laguerre nodes as a starting point (these are zeros of L_n)
        # Actually, for the Cohn-Goncalves construction, we want roots that
        # minimize the ratio, which relates to orthogonal polynomial properties
        # Start with spread-out positive values
        best_roots = np.array([4.0 * (i + 1) for i in range(12)])
    except:
        best_roots = np.array([4.0 * (i + 1) for i in range(12)])
    
    best_c4 = _compute_c4_fast(best_roots)
    
    # Try optimization
    try:
        # Use log-parameterization to ensure positivity
        def _log_transform(y):
            return np.exp(y)
        
        def _objective(y):
            roots = _log_transform(y)
            return _compute_c4_fast(roots)
        
        # Initial guess: log of default roots
        y0 = np.log(best_roots)
        
        # Multiple restarts with different strategies
        n_restarts = 8
        for r in range(n_restarts):
            if time.time() - _start_time > _DEADLINE - 90:
                break
            
            if r == 0:
                y_init = y0.copy()
            elif r == 1:
                # Try compressed roots
                np.random.seed(100 + r)
                y_init = y0 + np.random.randn(12) * 0.3
            elif r == 2:
                # Try expanded roots
                np.random.seed(200 + r)
                y_init = y0 + np.random.randn(12) * 0.8
            else:
                # Random perturbations
                np.random.seed(300 + r)
                y_init = y0 + np.random.randn(12) * (0.2 + 0.1 * r)
            
            try:
                result = minimize(_objective, y_init, method='L-BFGS-B', 
                                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if result.fun < best_c4:
                    roots = _log_transform(result.x)
                    if np.all(roots > 0) and np.all(np.diff(roots) > 0):
                        c4 = _compute_c4_fast(roots)
                        if c4 < best_c4:
                            best_c4 = c4
                            best_roots = roots.copy()
            except Exception:
                continue
            
            # Check time
            if time.time() - _start_time > _DEADLINE - 120:
                break
        
        # Final high-precision polish pass with tighter tolerances
        if time.time() - _start_time < _DEADLINE - 60:
            try:
                def _objective_precise(y):
                    roots = _log_transform(y)
                    return _compute_c4(roots)  # Use the more accurate version
                
                y_init = np.log(best_roots)
                result = minimize(_objective_precise, y_init, method='L-BFGS-B',
                                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
                
                if result.fun < best_c4:
                    roots = _log_transform(result.x)
                    if np.all(roots > 0) and np.all(np.diff(roots) > 0):
                        c4 = _compute_c4(roots)
                        if c4 < best_c4:
                            best_c4 = c4
                            best_roots = roots.copy()
            except Exception:
                pass
    except Exception:
        pass
    
    return list(best_roots)
# EVOLVE-BLOCK-END
