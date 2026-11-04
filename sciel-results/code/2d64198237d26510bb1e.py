"""
Refined optimization for C4 upper bound with 12 double Laguerre roots.
Uses log-space parameterization to enforce positivity, efficient moment
computation, and multi-stage optimization with analytical gradients.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_laguerre, roots_genlaguerre

# Precompute factorials up to degree 48 (max needed for P^2 where P has degree 24)
_FACTS = np.array([math.factorial(k) for k in range(50)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly (ascending coeff) * e^{-x} dx = sum_k coeffs[k]*k!."""
    n = min(len(coeffs), len(_FACTS))
    if n == 0:
        return 0.0
    return float(np.dot(coeffs[:n], _FACTS[:n]))

def _build_p_coeffs(roots):
    """Build P(x) = prod (x-x_i)^2 in ascending coeff order. Returns P_coeffs (degree 24)."""
    from numpy.polynomial.polynomial import polyfromroots, polymul
    B = polyfromroots(roots)  # degree 12, ascending
    P = polymul(B, B)          # degree 24
    return P

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2/I1^2 where I1=int P e^{-x}, I2=int P^2 e^{-x}, P=prod(x-x_i)^2."""
    roots = np.asarray(roots, dtype=np.float64)
    n = len(roots)
    if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    P = _build_p_coeffs(roots)
    I1 = _poly_moment(P)
    if I1 <= 0 or not np.isfinite(I1):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    from numpy.polynomial.polynomial import polymul
    P2 = polymul(P, P)
    I2 = _poly_moment(P2)
    if not np.isfinite(I2):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    c4 = I2 / (I1 * I1)
    if not need_grad:
        return float(c4)

    # Efficient gradient computation
    grad = np.zeros(n)
    inv_I1 = 1.0 / I1
    inv_I1_sq = inv_I1 * inv_I1
    inv_I1_cu = inv_I1_sq * inv_I1

    for i in range(n):
        r = roots[i]
        P_arr = P
        deg = len(P_arr) - 1
        S_i = np.zeros(deg)  # degree 23
        carry = P_arr[deg]
        for k in range(deg - 1, -1, -1):
            S_i[k] = carry
            carry = P_arr[k] + carry * r
        
        dI1_i = -2.0 * _poly_moment(S_i)
        T_i = polymul(P, S_i)
        dI2_i = -4.0 * _poly_moment(T_i)
        
        grad[i] = dI2_i * inv_I1_sq - 2.0 * I2 * dI1_i * inv_I1_cu

    return float(c4), grad

def _log_objective(log_roots):
    """Objective in log-space: roots = exp(log_roots)."""
    roots = np.exp(log_roots)
    return _compute_c4(roots)

def _log_objective_grad(log_roots):
    """Objective and gradient in log-space."""
    roots = np.exp(log_roots)
    c4, grad = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(log_roots)
    # Chain rule: dC4/d(log_r_i) = dC4/dr_i * r_i
    log_grad = grad * roots
    return c4, log_grad

def construct():
    start = time.time()
    DEADLINE = 600.0
    SAFETY = 30.0
    deadline = start + DEADLINE - SAFETY

    def time_ok():
        return (time.time() - start) < deadline

    best_roots = None
    best_c4 = np.inf

    # Phase 1: Grid search over generalized Laguerre parameter alpha
    for alpha in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]:
        if not time_ok():
            break
        try:
            xg, _ = roots_genlaguerre(12, alpha)
            xg = np.sort(xg)
            if np.all(xg > 0) and np.all(np.isfinite(xg)):
                c4g = _compute_c4(xg)
                if c4g < best_c4:
                    best_c4 = c4g
                    best_roots = xg.copy()
        except Exception:
            pass

    # Standard Laguerre as fallback
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    c4_0 = _compute_c4(x0)
    if c4_0 < best_c4:
        best_c4 = c4_0
        best_roots = x0.copy()

    # Phase 2: Differential evolution in log-space
    if time_ok() and best_roots is not None:
        log_bounds = [(math.log(1e-4), math.log(200.0))] * 12
        try:
            de_result = differential_evolution(
                _log_objective,
                bounds=log_bounds,
                seed=42,
                maxiter=300,
                popsize=20,
                tol=1e-12,
                polish=False,
                mutation=(0.4, 1.0),
                recombination=0.7,
                update_interval=10
            )
            if np.isfinite(de_result.fun):
                candidate = np.sort(np.exp(de_result.x))
                if np.all(candidate > 0):
                    c4 = _compute_c4(candidate)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
        except Exception:
            pass

    # Second DE run with different seed
    if time_ok():
        try:
            de_result2 = differential_evolution(
                _log_objective,
                bounds=log_bounds,
                seed=137,
                maxiter=200,
                popsize=18,
                tol=1e-12,
                polish=False,
                mutation=(0.5, 1.0),
                recombination=0.6,
                update_interval=12
            )
            if np.isfinite(de_result2.fun):
                candidate = np.sort(np.exp(de_result2.x))
                if np.all(candidate > 0):
                    c4 = _compute_c4(candidate)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 3: L-BFGS-B refinement in log-space from multiple starting points
    candidates_to_refine = [np.log(best_roots.copy())]
    
    rng = np.random.default_rng(42)
    for scale in [0.05, 0.1, 0.2, 0.35]:
        for _ in range(3):
            pert = np.log(best_roots) + scale * rng.standard_normal(12)
            candidates_to_refine.append(pert)

    # Add best alpha-derived roots in log-space
    for alpha in [0.5, 1.0, 2.0]:
        try:
            xg, _ = roots_genlaguerre(12, alpha)
            xg = np.sort(xg)
            if np.all(xg > 0):
                candidates_to_refine.append(np.log(xg.copy()))
        except Exception:
            pass

    for init in candidates_to_refine:
        if not time_ok():
            break
        try:
            res = minimize(
                _log_objective_grad,
                init,
                jac=True,
                method='L-BFGS-B',
                bounds=[(-20, 5)] * 12,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            if np.isfinite(res.fun):
                candidate = np.sort(np.exp(res.x))
                if np.all(candidate > 0):
                    c4 = _compute_c4(candidate)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 4: High-precision polish passes
    for polish_iter in range(8):
        if not time_ok():
            break
        try:
            log_best = np.log(best_roots)
            res = minimize(
                _log_objective_grad,
                log_best,
                jac=True,
                method='L-BFGS-B',
                bounds=[(-25, 10)] * 12,
                options={'maxiter': 5000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 100}
            )
            if np.isfinite(res.fun):
                candidate = np.sort(np.exp(res.x))
                if np.all(candidate > 0):
                    c4 = _compute_c4(candidate)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
                    else:
                        break
        except Exception:
            break

    # Phase 5: Coordinate descent refinement in log-space
    if time_ok():
        for cd_iter in range(300):
            if not time_ok():
                break
            improved = False
            log_roots = np.log(best_roots)
            for i in range(12):
                base = log_roots.copy()
                for delta in [1e-7, -1e-7, 1e-6, -1e-6, 1e-5, -1e-5, 1e-4, -1e-4, 1e-3, -1e-3]:
                    test = base.copy()
                    test[i] += delta
                    roots_test = np.exp(test)
                    if np.all(roots_test > 0) and np.all(np.isfinite(roots_test)):
                        c4_test = _compute_c4(roots_test)
                        if c4_test < best_c4:
                            best_c4 = c4_test
                            best_roots = np.sort(roots_test)
                            improved = True
                            break
            if not improved:
                break

    # Final safety check
    if best_roots is None or not np.all(np.isfinite(best_roots)) or not np.all(best_roots > 0):
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)

    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
