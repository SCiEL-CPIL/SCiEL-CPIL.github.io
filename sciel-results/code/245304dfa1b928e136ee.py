import numpy as np
import time
from scipy.special import genlaguerre, gammaln
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 600.0
_SAFETY_MARGIN = 30.0

def _deadline_reached():
    return (time.time() - _START_TIME) >= (_DEADLINE - _SAFETY_MARGIN)

def _compute_c4_bound(roots, m=12, n=1, num_quad=400):
    """
    Compute the C4 uncertainty upper bound for given double-root locations.
    C4 = (∫₀^∞ |P(x)|² x^(2n-1) e^(-x) dx)^(1/2) / (∫₀^∞ x^(2n-1) e^(-x) dx)^(1/2)
    where P(x) = ∏ᵢ (x - rᵢ)²
    """
    try:
        x_quad, w_quad = np.polynomial.laguerre.laggauss(num_quad)
        
        log_abs_P_sq = np.zeros_like(x_quad)
        for r in roots:
            log_abs_P_sq += 4.0 * np.log(np.abs(x_quad - r) + 1e-300)
        
        alpha = 2 * n - 1
        # weight is x^alpha e^-x
        # log weight = alpha log(x) - x
        if alpha == 1:
            log_weight = np.log(x_quad) - x_quad
        else:
            log_weight = alpha * np.log(x_quad) - x_quad
            
        total_log = log_abs_P_sq + log_weight
        
        max_log = np.max(total_log)
        if np.isfinite(max_log):
            integrand_scaled = np.exp(total_log - max_log)
            integral_val = np.sum(w_quad * integrand_scaled) * np.exp(max_log)
        else:
            return np.inf
            
        log_norm = 0.5 * gammaln(2 * n)
        norm = np.exp(log_norm)
        
        c4_bound = np.sqrt(integral_val / norm)
        return c4_bound
    except Exception:
        return np.inf

def _objective(roots_flat, m=12, n=1):
    """Objective function for optimization: the C4 bound."""
    roots = np.asarray(roots_flat)
    if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
        return 1e10
    return _compute_c4_bound(roots, m, n)

def _get_initial_roots(m=12, n=1):
    """
    Get initial root positions.
    Using roots of L_m^(2n-1)(x) as they are orthogonal w.r.t the weight in the integral.
    """
    try:
        alpha = 2 * n - 1
        L = genlaguerre(m, alpha)
        roots = L.roots()
        roots = np.real(roots)
        roots = roots[roots > 0]
        roots = np.sort(roots)
        if len(roots) >= m:
            return roots[:m]
        else:
            return np.linspace(1.0, 20.0, m)
    except Exception:
        return np.linspace(1.0, 20.0, m)

def construct():
    """
    Construct the 12 positive double-root locations for the Cohn-Goncalves
    Laguerre polynomial construction with m=12, n=1.
    """
    m = 12
    n = 1
    
    # Get initial roots from Laguerre polynomial L_m^(2n-1)
    roots_init = _get_initial_roots(m, n)
    roots_init = np.maximum(roots_init, 1e-6)
    roots_init = np.sort(roots_init)
    
    best_roots = roots_init.copy()
    best_score = _objective(roots_init, m, n)
    
    # Strategy 1: Nelder-Mead from Laguerre roots
    if not _deadline_reached():
        try:
            result = minimize(
                _objective,
                roots_init,
                args=(m, n),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11}
            )
            if result.fun < best_score:
                best_roots = result.x
                best_score = result.fun
        except Exception:
            pass

    # Strategy 2: Try starting from roots of L_m(x) (alpha=0)
    if not _deadline_reached():
        try:
            L0 = genlaguerre(m, 0)
            roots_alt = np.real(L0.roots())
            roots_alt = roots_alt[roots_alt > 0]
            roots_alt = np.sort(roots_alt)
            if len(roots_alt) == m:
                result_alt = minimize(
                    _objective,
                    roots_alt,
                    args=(m, n),
                    method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-11}
                )
                if result_alt.fun < best_score:
                    best_roots = result_alt.x
                    best_score = result_alt.fun
        except Exception:
            pass

    # Strategy 3: Local Refinement
    if not _deadline_reached():
        try:
            result_refine = minimize(
                _objective,
                best_roots,
                args=(m, n),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-11, 'fatol': 1e-13}
            )
            if result_refine.fun < best_score:
                best_roots = result_refine.x
                best_score = result_refine.fun
        except Exception:
            pass

    # Ensure roots are positive and sorted
    best_roots = np.maximum(best_roots, 1e-6)
    best_roots = np.sort(best_roots)
    
    return [float(x) for x in best_roots]

# EVOLVE-BLOCK-END
