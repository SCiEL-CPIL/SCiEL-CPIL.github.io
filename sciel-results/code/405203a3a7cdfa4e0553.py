"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, root
from scipy.special import roots_laguerre
import time

def _compute_c4_bound_fast(roots):
    """Compute the C4 upper bound for given double-root positions using Gauss-Laguerre quadrature."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Vectorized: compute log_P for all roots at once
    diff = x_quad[:, np.newaxis] - roots[np.newaxis, :]
    log_abs_diff = np.log(np.abs(diff) + 1e-300)
    log_P = 2.0 * np.sum(log_abs_diff, axis=1)
    
    P = np.exp(log_P)
    P2 = P * P
    
    int_P = np.sum(w_quad * P)
    int_P2 = np.sum(w_quad * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
        return np.inf
    
    c4 = int_P2 / (int_P * int_P)
    return c4

def _compute_c4_bound(roots):
    """Alias for compatibility."""
    return _compute_c4_bound_fast(roots)

def _objective(roots):
    """Objective function for optimization."""
    return _compute_c4_bound_fast(roots)

def _rational_potential_residuals(params):
    """
    Residuals for the rational potential parameterization.
    params: [a1, a2, a3, c1, c2, c3, x1, ..., x12]
    We want the roots x_i to satisfy the equilibrium condition:
    sum_{j != i} 1/(x_i - x_j) + V'(x_i) = 0
    where V(x) = sum_{k=1}^3 a_k / (x - c_k)
    and V'(x) = -sum_{k=1}^3 a_k / (x - c_k)^2
    
    We have 12 equations (one for each root) and 18 unknowns (6 potential params + 12 roots).
    To make this a square system, we can fix 6 of the roots or use a different approach.
    
    Alternative: Optimize the 6 potential parameters to minimize the sum of squared residuals
    of the equilibrium conditions for a given set of roots, then refine the roots.
    
    Actually, let's use a different structural approach:
    We parameterize the 12 roots as x_i = f(t_i) where t_i are 12 parameters,
    but we constrain them to be critical points of a potential with 3 poles.
    
    This is complex. Let's try a simpler structural change:
    Use the fact that for Laguerre polynomials, the roots are related to the zeros
    of orthogonal polynomials. The "double root" structure suggests we are looking
    for a polynomial whose critical points are the roots of another polynomial.
    
    Let's try: P(x) = prod (x - r_i)^2. The critical points of P are the roots of P'(x).
    P'(x) = 2 * prod (x - r_i) * sum_i prod_{j != i} (x - r_j)
    So the critical points are the r_i themselves (double roots become single critical points? No.)
    
    Actually, if P(x) = prod (x - r_i)^2, then P'(x) = 2 prod (x - r_i) sum_i prod_{j!=i} (x-r_j).
    The zeros of P' are the r_i (each with multiplicity 1) plus the zeros of sum_i prod_{j!=i} (x-r_j).
    The sum term is P'(x) / (2 prod (x-r_i)).
    
    This is getting complicated. Let's stick to the rational potential idea but implement it
    as a two-step optimization:
    1. Given 12 roots, find the best 3-pole potential that approximates the equilibrium.
    2. Given the potential, find the 12 critical points.
    3. Iterate.
    
    For now, let's use a different structural change: 
    Use the "log-gas" equilibrium equation directly. The roots r_i satisfy:
    sum_{j != i} 1/(r_i - r_j) = 1/(2 r_i) - sum_{j} 1/(r_i - r_j) ... no.
    
    The equilibrium condition for Laguerre zeros is:
    sum_{j != i} 1/(r_i - r_j) = 1/(2 r_i)
    This is for the standard Laguerre polynomial L_n(x).
    
    For our problem, we have double roots, so the potential is different.
    Let's assume the equilibrium condition is:
    sum_{j != i} 1/(r_i - r_j) = V'(r_i)
    where V is some effective potential.
    
    Let's try a new approach: 
    Use the fact that the problem is scale-invariant? No, the weight is e^{-x}, so it's not scale-invariant.
    
    Let's try a different parameterization: 
    r_i = a * i + b * i^2 + c * i^3 + d
    This is a cubic polynomial in i. 4 parameters instead of 12.
    This enforces smooth, non-geometric root density.
    """
    # This function is a placeholder for the structural change.
    # We will implement the cubic parameterization in construct().
    return 0.0

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 580  # Safety margin before 630s kill
    
    # Start from standard Laguerre roots as initial guess
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound_fast(best_roots)
    
    rng = np.random.default_rng(12345)
    
    # STRUCTURAL CHANGE: 
    # Parameterize the 12 roots using a low-dimensional model.
    # We try several different parameterizations and optimize the parameters.
    
    # Parameterization 1: Cubic in index
    # r_i = a0 + a1*i + a2*i^2 + a3*i^3
    def _cubic_objective(params):
        a0, a1, a2, a3 = params
        i = np.arange(12)
        roots = a0 + a1*i + a2*i**2 + a3*i**3
        if np.any(roots <= 0) or np.any(np.isnan(roots)):
            return 1e10
        return _compute_c4_bound_fast(roots)
    
    # Multistart for cubic
    for a0_init in [0.1, 0.5, 1.0, 2.0]:
        for a1_init in [0.5, 1.0, 2.0]:
            for a2_init in [0.0, 0.1, 0.2]:
                for a3_init in [0.0, 0.01, 0.05]:
                    if time.time() > deadline - 60:
                        break
                    try:
                        result = minimize(
                            _cubic_objective,
                            np.array([a0_init, a1_init, a2_init, a3_init]),
                            method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                        )
                        if result.fun < best_score:
                            a0, a1, a2, a3 = result.x
                            i = np.arange(12)
                            roots = a0 + a1*i + a2*i**2 + a3*i**3
                            if np.all(roots > 0):
                                candidate = np.sort(roots)
                                score = _compute_c4_bound_fast(candidate)
                                if score < best_score:
                                    best_score = score
                                    best_roots = candidate.copy()
                    except Exception:
                        pass
                if time.time() > deadline - 60:
                    break
            if time.time() > deadline - 60:
                break
        if time.time() > deadline - 60:
            break
    
    # Parameterization 2: Log-cubic
    # log(r_i) = a0 + a1*i + a2*i^2 + a3*i^3
    def _log_cubic_objective(params):
        a0, a1, a2, a3 = params
        i = np.arange(12)
        log_roots = a0 + a1*i + a2*i**2 + a3*i**3
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(roots > 1e6):
            return 1e10
        return _compute_c4_bound_fast(roots)
    
    for a0_init in [-2, -1, 0, 1]:
        for a1_init in [0.1, 0.3, 0.5]:
            for a2_init in [0.0, 0.05, 0.1]:
                for a3_init in [0.0, 0.01, 0.05]:
                    if time.time() > deadline - 60:
                        break
                    try:
                        result = minimize(
                            _log_cubic_objective,
                            np.array([a0_init, a1_init, a2_init, a3_init]),
                            method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                        )
                        if result.fun < best_score:
                            a0, a1, a2, a3 = result.x
                            i = np.arange(12)
                            log_roots = a0 + a1*i + a2*i**2 + a3*i**3
                            roots = np.exp(log_roots)
                            if np.all(roots > 0) and np.all(roots < 1e6):
                                candidate = np.sort(roots)
                                score = _compute_c4_bound_fast(candidate)
                                if score < best_score:
                                    best_score = score
                                    best_roots = candidate.copy()
                    except Exception:
                        pass
                if time.time() > deadline - 60:
                    break
            if time.time() > deadline - 60:
                break
        if time.time() > deadline - 60:
            break
    
    # Refine the best result with full 12-dimensional optimization
    if time.time() < deadline - 30:
        try:
            result_final = minimize(
                _objective,
                best_roots,
                method='L-BFGS-B',
                bounds=[(1e-8, None)] * 12,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if result_final.fun < best_score:
                candidate = np.sort(result_final.x)
                if np.all(candidate > 0):
                    score = _compute_c4_bound_fast(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
        except Exception:
            pass
    
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
