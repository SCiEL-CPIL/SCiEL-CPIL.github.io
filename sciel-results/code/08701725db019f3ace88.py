import numpy as np
from scipy.optimize import minimize, minimize_scalar, root
from scipy.special import roots_laguerre, roots_genlaguerre
from numpy.polynomial.polynomial import polyfromroots, polymul
import time
import math

_FACTS = np.array([math.factorial(k) for k in range(80)], dtype=np.float64)

def _poly_moment(coeffs):
    n = min(len(coeffs), len(_FACTS))
    if n == 0:
        return 0.0
    return float(np.dot(coeffs[:n], _FACTS[:n]))

def _compute_c4(roots, need_grad=False):
    roots = np.asarray(roots, dtype=np.float64)
    n = len(roots)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    B = polyfromroots(roots)
    P = polymul(B, B)
    P2 = polymul(P, P)

    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf
    c4 = I2 / (I1 * I1)
    if not need_grad:
        return float(c4)

    grad = np.zeros_like(roots)
    B2 = polymul(B, B)
    B3 = polymul(B2, B)
    
    for i in range(n):
        R_i = np.array([1.0])
        for j in range(n):
            if j == i:
                continue
            R_i = polymul(R_i, np.array([1.0, -roots[j]]))
        
        BR_i = polymul(B, R_i)
        dI1_i = -2.0 * _poly_moment(BR_i)
        
        B3R_i = polymul(B3, R_i)
        dI2_i = -4.0 * _poly_moment(B3R_i)
        
        grad[i] = dI2_i / (I1 * I1) - 2.0 * I2 * dI1_i / (I1 ** 3)
    
    return float(c4), grad

def _objective(roots):
    return _compute_c4(roots)

def _objective_grad(roots):
    c4, g = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(roots)
    return c4, g

def _c4_from_alpha(alpha):
    try:
        xg, _ = roots_genlaguerre(12, alpha)
        xg = np.sort(xg)
        if not np.all(xg > 0) or not np.isfinite(xg).all():
            return np.inf
        return _compute_c4(xg)
    except Exception:
        return np.inf

def _calogero_equations(x, a, b):
    """Equilibrium conditions for Calogero-Sutherland model.
    For each i: a + b/x_i - sum_{j!=i} 1/(x_i - x_j)^2 = 0
    """
    n = len(x)
    eqs = np.zeros(n)
    for i in range(n):
        eqs[i] = a + b / x[i]
        for j in range(n):
            if j != i:
                eqs[i] -= 1.0 / (x[i] - x[j]) ** 2
    return eqs

def _calogero_jacobian(x, a, b):
    """Jacobian of equilibrium equations w.r.t. x."""
    n = len(x)
    J = np.zeros((n, n))
    for i in range(n):
        J[i, i] = -b / x[i]**2
        for j in range(n):
            if j != i:
                J[i, j] = 2.0 / (x[i] - x[j])**3
    return J

def _solve_calogero(a, b, x0):
    """Solve Calogero-Sutherland equilibrium for given a, b using Newton-Raphson."""
    x = np.array(x0, dtype=np.float64)
    n = len(x)
    
    for iteration in range(100):
        eqs = _calogero_equations(x, a, b)
        if np.max(np.abs(eqs)) < 1e-12:
            break
        try:
            J = _calogero_jacobian(x, a, b)
            dx = np.linalg.solve(J, -eqs)
            x = x + dx
            x = np.sort(x)
            if np.any(x <= 0):
                x = np.maximum(x, 1e-10)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            try:
                dx = np.linalg.lstsq(J, -eqs, rcond=None)[0]
                x = x + 0.5 * dx
                x = np.sort(x)
                if np.any(x <= 0):
                    x = np.maximum(x, 1e-10)
            except:
                break
    
    # Verify solution
    eqs = _calogero_equations(x, a, b)
    if np.max(np.abs(eqs)) > 1e-8 or np.any(x <= 0):
        return None
    return x

def _c4_from_calogero(a, b, x0):
    """Compute C4 for Calogero-Sutherland equilibrium roots."""
    try:
        x = _solve_calogero(a, b, x0)
        if x is None:
            return np.inf
        return _compute_c4(x)
    except Exception:
        return np.inf

def _calogero_objective(params, x0):
    """Objective for 2-parameter Calogero optimization."""
    log_a, log_b = params
    a = np.exp(log_a)
    b = np.exp(log_b)
    if a < 0 or b < 0:
        return np.inf
    return _c4_from_calogero(a, b, x0)

def _calogero_objective_grad(params, x0):
    """Objective with gradient for 2-parameter Calogero optimization."""
    log_a, log_b = params
    a = np.exp(log_a)
    b = np.exp(log_b)
    
    x = _solve_calogero(a, b, x0)
    if x is None:
        return np.inf, np.zeros(2)
    
    c4 = _compute_c4(x)
    if not np.isfinite(c4):
        return np.inf, np.zeros(2)
    
    # Numerical gradient w.r.t. log_a and log_b
    eps = 1e-7
    c4_a = _c4_from_calogero(a * (1 + eps), b, x0)
    c4_b = _c4_from_calogero(a, b * (1 + eps), x0)
    
    grad_a = (c4_a - c4) / (a * eps) if np.isfinite(c4_a) else 0.0
    grad_b = (c4_b - c4) / (b * eps) if np.isfinite(c4_b) else 0.0
    
    return c4, np.array([grad_a, grad_b])

def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 45.0
    
    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    # Phase 1: 1D search over alpha to find optimal root density profile
    best_alpha = 0.0
    best_c4_alpha = np.inf
    
    log_alphas = np.linspace(math.log(1e-3), math.log(1e4), 300)
    for la in log_alphas:
        if not time_ok():
            break
        c4_val = _c4_from_alpha(np.exp(la))
        if c4_val < best_c4_alpha:
            best_c4_alpha = c4_val
            best_alpha = np.exp(la)

    if time_ok():
        try:
            res_alpha = minimize_scalar(
                lambda la: _c4_from_alpha(np.exp(la)),
                bounds=(math.log(max(best_alpha * 0.5, 1e-4)), math.log(min(best_alpha * 2.0, 1e5))),
                method='bounded',
                options={'xatol': 1e-10, 'maxiter': 500}
            )
            if res_alpha.success and np.isfinite(res_alpha.fun) and res_alpha.fun < best_c4_alpha:
                best_alpha = np.exp(res_alpha.x)
                best_c4_alpha = res_alpha.fun
        except Exception:
            pass

    xg, _ = roots_genlaguerre(12, best_alpha)
    best_roots = np.sort(xg)
    best_c4 = _compute_c4(best_roots)

    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    c4_0 = _compute_c4(x0)
    if c4_0 < best_c4:
        best_c4 = c4_0
        best_roots = x0.copy()

    # Phase 2: Calogero-Sutherland 2-parameter optimization
    # Use best_roots as initial guess for x
    if time_ok():
        # Grid search over log_a, log_b
        best_calogero_params = None
        best_calogero_c4 = np.inf
        
        log_a_grid = np.linspace(math.log(1e-2), math.log(1e2), 30)
        log_b_grid = np.linspace(math.log(1e-2), math.log(1e2), 30)
        
        for la in log_a_grid:
            if not time_ok():
                break
            for lb in log_b_grid:
                if not time_ok():
                    break
                a = np.exp(la)
                b = np.exp(lb)
                c4_val = _c4_from_calogero(a, b, best_roots)
                if c4_val < best_calogero_c4:
                    best_calogero_c4 = c4_val
                    best_calogero_params = (la, lb)
        
        # Refine with 2D minimization
        if best_calogero_params is not None and time_ok():
            try:
                res_cal = minimize(
                    lambda p: _calogero_objective(p, best_roots),
                    best_calogero_params,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if np.isfinite(res_cal.fun) and res_cal.fun < best_calogero_c4:
                    best_calogero_c4 = res_cal.fun
                    best_calogero_params = res_cal.x
            except Exception:
                pass
        
        # Get final roots from best Calogero parameters
        if best_calogero_params is not None and time_ok():
            a_final = np.exp(best_calogero_params[0])
            b_final = np.exp(best_calogero_params[1])
            x_cal = _solve_calogero(a_final, b_final, best_roots)
            if x_cal is not None:
                c4_cal = _compute_c4(x_cal)
                if c4_cal < best_c4:
                    best_c4 = c4_cal
                    best_roots = x_cal.copy()

    # Phase 3: Full 12-DOF L-BFGS-B refinement with analytic gradient
    if time_ok():
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-8, None)] * 12,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 4: Multi-start with perturbed initializations
    if time_ok():
        rng = np.random.default_rng(42)
        for trial in range(8):
            if not time_ok():
                break
            pert = best_roots * (1.0 + 0.2 * rng.standard_normal(12))
            pert = np.maximum(pert, 1e-3)
            pert = np.sort(pert)
            try:
                res = minimize(
                    _objective_grad,
                    pert,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(1e-8, None)] * 12,
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40}
                )
                if np.all(res.x > 0) and np.isfinite(res.fun):
                    candidate = np.sort(res.x)
                    c4 = _compute_c4(candidate)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
            except Exception:
                pass

    # Phase 5: Try neighboring alpha values
    if time_ok():
        for alpha_offset in [0.5, 2.0, 0.1, 5.0, 0.2, 0.8]:
            if not time_ok():
                break
            for sign in [-1, 1]:
                a_try = best_alpha * alpha_offset if sign > 0 else best_alpha / alpha_offset
                if a_try < 1e-3 or a_try > 1e4:
                    continue
                try:
                    xg2, _ = roots_genlaguerre(12, a_try)
                    xg2 = np.sort(xg2)
                    if not np.all(xg2 > 0):
                        continue
                    res2 = minimize(
                        _objective_grad,
                        xg2,
                        jac=True,
                        method='L-BFGS-B',
                        bounds=[(1e-8, None)] * 12,
                        options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40}
                    )
                    if np.all(res2.x > 0) and np.isfinite(res2.fun):
                        candidate = np.sort(res2.x)
                        c4 = _compute_c4(candidate)
                        if c4 < best_c4:
                            best_c4 = c4
                            best_roots = candidate.copy()
                except Exception:
                    pass

    # Final safety check
    if time.time() > DEADLINE - 5:
        return best_roots.tolist()
    
    return best_roots.tolist()
