"""Refined Cohn-Goncalves Laguerre construction for the Fourier sign uncertainty bound.

We tune the 12 positive double-root locations (m=12, n=1) to minimize the
upper bound C4 on the Fourier sign uncertainty.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_laguerre

_DEADLINE = 560.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _seed_roots():
    return [4.0 * (i + 1) for i in range(12)]


def _valid(roots):
    """Root constraints: 12 positive, strictly increasing (distinct) roots."""
    if len(roots) != 12:
        return False
    if np.any(roots <= 0.0):
        return False
    if np.any(np.diff(roots) <= 0.0):
        return False
    return True


def _compute_c4(roots):
    """
    Compute the C4 upper bound for the Fourier sign uncertainty using the
    Cohn-Goncalves Laguerre construction with double roots.
    
    For double roots r_1 < ... < r_m, the polynomial is Q(x) = prod_j (x - r_j)^2.
    The C4 bound is the ratio of the L2 norm squared of Q to the L2 norm squared
    of Q' in the Laguerre inner product <f,g> = int_0^inf f(x)g(x) e^{-x} dx.
    """
    if not _valid(roots):
        return 1e10
    
    r = np.array(roots, dtype=float)
    m = len(r)
    
    # Use Gauss-Laguerre quadrature
    n_quad = max(120, 3 * m)
    x_quad, w_quad = roots_laguerre(n_quad)
    
    # Compute Q(x) = prod_j (x - r_j)^2 in log space
    log_q = np.zeros(n_quad)
    for rj in r:
        log_q += 2.0 * np.log(np.abs(x_quad - rj) + 1e-300)
    q_vals = np.exp(log_q)
    
    # Q'(x) = Q(x) * sum_j 2/(x - r_j)
    q_prime_factor = np.zeros(n_quad)
    for rj in r:
        q_prime_factor += 2.0 / (x_quad - rj)
    q_prime_vals = q_vals * q_prime_factor
    
    # L2 norms with weight e^{-x} (included in quadrature weights)
    norm_q = np.sum(w_quad * q_vals**2)
    norm_q_prime = np.sum(w_quad * q_prime_vals**2)
    
    if norm_q_prime < 1e-30 or norm_q < 0:
        return 1e10
    
    c4 = norm_q / norm_q_prime
    return c4


def construct():
    if _elapsed() > _DEADLINE:
        return _seed_roots()

    try:
        best_score = 1e10
        best_roots = None
        
        # Multi-start: arithmetic progressions with various scales
        for s in [5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 50.0]:
            roots = [4.0 * s * (i + 1) for i in range(12)]
            if _valid(roots):
                score = _compute_c4(roots)
                if score < best_score:
                    best_score = score
                    best_roots = list(roots)
        
        # Multi-start: scaled Laguerre zeros
        try:
            lag_zeros = roots_laguerre(12)[0]
            lag_zeros = np.sort(lag_zeros)
            for target_max in [50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]:
                scale = target_max / lag_zeros[-1]
                roots = list(scale * lag_zeros)
                if _valid(roots):
                    score = _compute_c4(roots)
                    if score < best_score:
                        best_score = score
                        best_roots = roots
        except Exception:
            pass
        
        # Multi-start: geometric progressions
        for base in [1.05, 1.1, 1.2, 1.5, 2.0]:
            for start in [5.0, 20.0, 50.0, 100.0]:
                roots = [start * base ** i for i in range(12)]
                if _valid(roots):
                    score = _compute_c4(roots)
                    if score < best_score:
                        best_score = score
                        best_roots = list(roots)
        
        # Local refinement with L-BFGS-B
        if best_roots is not None and _elapsed() < _DEADLINE - 120:
            x0 = np.array(best_roots, dtype=float)
            
            def objective(params):
                roots = np.sort(params)
                if not _valid(roots):
                    return 1e10
                return _compute_c4(roots)
            
            bounds = [(1e-3, 1e6)] * 12
            
            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'maxfun': 20000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if _elapsed() < _DEADLINE - 60:
                    if _valid(result.x):
                        refined = np.sort(result.x)
                        refined_score = _compute_c4(refined)
                        if refined_score < best_score:
                            best_score = refined_score
                            best_roots = list(refined)
            except Exception:
                pass
            
            # Second refinement with Nelder-Mead
            if best_roots is not None and _elapsed() < _DEADLINE - 60:
                x0 = np.array(best_roots, dtype=float)
                try:
                    result2 = minimize(
                        objective,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    if _valid(result2.x):
                        refined2 = np.sort(result2.x)
                        refined_score2 = _compute_c4(refined2)
                        if refined_score2 < best_score:
                            best_score = refined_score2
                            best_roots = list(refined2)
                except Exception:
                    pass
        
        if best_roots is not None and _valid(best_roots):
            return [float(r) for r in best_roots]
        
    except Exception:
        pass

    return _seed_roots()
