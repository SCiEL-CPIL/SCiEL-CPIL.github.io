"""Efficient 12-DOF optimization with generalized Laguerre multi-start and log-space polish.
Uses analytic gradients for C4 and L-BFGS-B with diverse initializations from L_12^(alpha).
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import roots_laguerre, roots_genlaguerre
from numpy.polynomial.polynomial import polyfromroots, polymul

_FACTS = np.array([math.factorial(k) for k in range(60)], dtype=np.float64)

def _poly_moment(coeffs):
    """int poly(x) e^{-x} dx over [0,inf) for ascending coefficient order."""
    n = len(coeffs)
    if n == 0:
        return 0.0
    m = min(n, 60)
    return float(np.dot(coeffs[:m], _FACTS[:m]))

def _build_p_squared_coeffs(roots):
    """Build P(x)=prod(x-x_i)^2 and P2=P^2 in ascending coeff order."""
    B = polyfromroots(roots)
    P = polymul(B, B)
    P2 = polymul(P, P)
    return P, P2

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2/I1^2 with optional gradient."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    P, P2 = _build_p_squared_coeffs(roots)
    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf
    c4 = I2 / (I1 * I1)
    if not need_grad:
        return float(c4)

    # Gradient via prefix/suffix products
    n = len(roots)
    prefix = [None] * n
    prefix[0] = np.ones(1)
    for j in range(n - 1):
        prefix[j + 1] = polymul(prefix[j], np.array([roots[j]**2, -2.0*roots[j], 1.0]))

    suffix = [None] * n
    suffix[n - 1] = np.ones(1)
    for j in range(n - 2, -1, -1):
        suffix[j] = polymul(suffix[j + 1], np.array([roots[j]**2, -2.0*roots[j], 1.0]))

    grad = np.zeros(n)
    for i in range(n):
        if i == 0:
            Q_i = suffix[0].copy()
        elif i == n - 1:
            Q_i = prefix[n - 1].copy()
        else:
            Q_i = polymul(prefix[i], suffix[i])

        int_Qi = _poly_moment(Q_i)
        xQi = polymul(Q_i, np.array([0.0, 1.0]))
        int_xQi = _poly_moment(xQi)
        int_P_over_xi = int_xQi - roots[i] * int_Qi
        dI1_i = 2.0 * int_P_over_xi

        S_i = polymul(Q_i, Q_i)
        x3Si = polymul(S_i, np.array([0.0, 0.0, 0.0, 1.0]))
        x2Si = polymul(S_i, np.array([0.0, 0.0, 1.0]))
        xSi = polymul(S_i, np.array([0.0, 1.0]))
        int_x3Si = _poly_moment(x3Si)
        int_x2Si = _poly_moment(x2Si)
        int_xSi = _poly_moment(xSi)
        int_Si = _poly_moment(S_i)
        dI2_i = 4.0 * (int_x3Si - 3.0*roots[i]*int_x2Si + 3.0*roots[i]**2*int_xSi - roots[i]**3*int_Si)

        grad[i] = dI2_i / (I1 * I1) - 2.0 * I2 * dI1_i / (I1 ** 3)

    return float(c4), grad

def _objective(roots):
    return _compute_c4(roots)

def _objective_grad(roots):
    c4, g = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(roots)
    return c4, g

def _exp_objective_grad(log_roots):
    """Objective in log-space: x_i = exp(log_roots_i), removes positivity constraint."""
    x = np.exp(log_roots)
    c4, grad_x = _compute_c4(x, need_grad=True)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(log_roots)
    grad_log = x * grad_x
    return c4, grad_log

def _c4_from_alpha(alpha):
    """Compute C4 for roots of generalized Laguerre polynomial L_12^(alpha)."""
    try:
        xg, _ = roots_genlaguerre(12, alpha)
        xg = np.sort(xg)
        if not np.all(xg > 0) or not np.isfinite(xg).all():
            return np.inf
        return _compute_c4(xg)
    except Exception:
        return np.inf


def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 45.0

    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    # Phase 1: Coarse-to-fine 1D search over alpha for generalized Laguerre
    best_alpha = 0.0
    best_c4_alpha = np.inf

    # Coarse grid
    log_alphas = np.linspace(math.log(1e-3), math.log(1e4), 60)
    for la in log_alphas:
        if not time_ok():
            break
        c4_val = _c4_from_alpha(np.exp(la))
        if c4_val < best_c4_alpha:
            best_c4_alpha = c4_val
            best_alpha = np.exp(la)

    # Fine grid around best
    if time_ok():
        local_lo = math.log(max(best_alpha * 0.5, 1e-3))
        local_hi = math.log(min(best_alpha * 2.0, 1e4))
        if local_hi > local_lo:
            log_alphas_fine = np.linspace(local_lo, local_hi, 250)
            for la in log_alphas_fine:
                if not time_ok():
                    break
                c4_val = _c4_from_alpha(np.exp(la))
                if c4_val < best_c4_alpha:
                    best_c4_alpha = c4_val
                    best_alpha = np.exp(la)

    # Scalar minimization
    if time_ok():
        try:
            res_alpha = minimize_scalar(
                lambda la: _c4_from_alpha(np.exp(la)),
                bounds=(math.log(1e-3), math.log(1e4)),
                method='bounded',
                options={'xatol': 1e-10, 'maxiter': 300}
            )
            if res_alpha.success and np.isfinite(res_alpha.fun) and res_alpha.fun < best_c4_alpha:
                best_alpha = np.exp(res_alpha.x)
                best_c4_alpha = res_alpha.fun
        except Exception:
            pass

    # Get base roots from best alpha
    xg, _ = roots_genlaguerre(12, best_alpha)
    best_roots = np.sort(xg)
    best_c4 = _compute_c4(best_roots)

    # Also try standard Laguerre
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    c4_0 = _compute_c4(x0)
    if c4_0 < best_c4:
        best_c4 = c4_0
        best_roots = x0.copy()

    # Phase 2: Multi-start L-BFGS-B with diverse alpha initializations
    # Generate a set of diverse initial points from generalized Laguerre roots
    initial_points = [best_roots.copy(), x0.copy()]
    
    # Add points from other alphas
    alphas_to_try = [best_alpha * 0.1, best_alpha * 0.5, best_alpha * 2.0, best_alpha * 10.0, 1.0, 0.1, 10.0]
    for a in alphas_to_try:
        if not time_ok():
            break
        try:
            if a > 0:
                xg_try, _ = roots_genlaguerre(12, a)
                xg_try = np.sort(xg_try)
                if np.all(xg_try > 0) and np.all(np.isfinite(xg_try)):
                    initial_points.append(xg_try)
        except Exception:
            pass

    # Add perturbed versions of the best root
    rng = np.random.default_rng(42)
    for _ in range(5):
        if not time_ok():
            break
        pert = best_roots * (1.0 + 0.3 * rng.standard_normal(12))
        pert = np.maximum(pert, 1e-3)
        pert = np.sort(pert)
        initial_points.append(pert)

    # Optimize each initial point
    for init_pt in initial_points:
        if not time_ok():
            break
        try:
            # Direct space optimization
            res = minimize(
                _objective_grad,
                init_pt,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-8, None)] * 12,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
            
            # Log-space optimization from this point
            if time_ok():
                log_init = np.log(np.maximum(init_pt, 1e-10))
                res_log = minimize(
                    _exp_objective_grad,
                    log_init,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 50}
                )
                if np.isfinite(res_log.fun):
                    candidate = np.sort(np.exp(res_log.x))
                    if np.all(candidate > 0) and np.all(np.isfinite(candidate)):
                        c4 = _compute_c4(candidate)
                        if c4 < best_c4:
                            best_c4 = c4
                            best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 3: Final high-precision polish
    for polish_iter in range(10):
        if not time_ok():
            break
        # Direct space polish
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-10, None)] * 12,
                options={'maxiter': 5000, 'ftol': 1e-17, 'gtol': 1e-16, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass
        # Log-space polish
        if not time_ok():
            break
        try:
            log_roots_init = np.log(best_roots)
            res_log = minimize(
                _exp_objective_grad,
                log_roots_init,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-17, 'gtol': 1e-16, 'maxls': 50}
            )
            if np.isfinite(res_log.fun):
                candidate = np.sort(np.exp(res_log.x))
                if np.all(candidate > 0) and np.all(np.isfinite(candidate)):
                    c4 = _compute_c4(candidate)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
        except Exception:
            pass

    # Fallback
    if not np.all(np.isfinite(best_roots)) or not np.all(best_roots > 0):
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)

    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
