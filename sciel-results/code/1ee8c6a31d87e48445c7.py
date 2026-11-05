"""Structural change: represent 12 double roots as equilibrium positions of a 1D gas
with repulsive 1/(x_i-x_j)^2 potential and logarithmic confining potential V(x)=a*ln(x).
The equilibrium conditions are 12 algebraic equations in 12 unknowns with one free parameter a.
We solve via Newton-Raphson for given a, then optimize a (1D) to minimize C4.
This replaces the 12-DOF discrete search with a 1-DOF continuous parameter search
over a smooth manifold of equilibrium configurations."""

import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar, newton
from scipy.special import roots_laguerre
from numpy.polynomial.polynomial import polyfromroots, polymul

_FACTS = np.array([math.factorial(k) for k in range(80)], dtype=np.float64)

def _poly_moment(coeffs):
    n = min(len(coeffs), len(_FACTS))
    if n == 0:
        return 0.0
    return float(np.dot(coeffs[:n], _FACTS[:n]))

def _compute_c4(roots):
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    B = polyfromroots(roots)
    P = polymul(B, B)
    P2 = polymul(P, P)
    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        return np.inf
    return float(I2 / (I1 * I1))

def _equilibrium_residuals(roots, a):
    """Residuals of equilibrium conditions: for each i,
    a/x_i + 2*sum_{j!=i} 1/(x_i - x_j) = 0
    (factor of 2 from double roots / symmetric repulsion)
    Actually the force balance for 1/(x_i-x_j)^2 potential:
    dV/dx_i = a/x_i - 2*sum_{j!=i} 1/(x_i-x_j)^2 ... 
    Let me reconsider: potential U = sum_{i<j} 1/(x_i-x_j)^2 + a*sum_i ln(x_i)
    Force on particle i: -dU/dx_i = -a/x_i + 2*sum_{j!=i} 1/(x_i-x_j)^3 = 0
    So: a/x_i = 2*sum_{j!=i} 1/(x_i-x_j)^3
    """
    n = len(roots)
    res = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if j == i:
                continue
            d = roots[i] - roots[j]
            s += 1.0 / (d * d * d)
        res[i] = a / roots[i] - 2.0 * s
    return res

def _equilibrium_jacobian(roots, a):
    """Jacobian of equilibrium residuals w.r.t. roots."""
    n = len(roots)
    J = np.zeros((n, n))
    for i in range(n):
        J[i, i] = -a / (roots[i] * roots[i])
        for j in range(n):
            if j == i:
                continue
            d = roots[i] - roots[j]
            J[i, j] = -6.0 / (d * d * d * d)
    return J

def _solve_equilibrium(a, initial_roots, max_iter=200, tol=1e-12):
    """Solve equilibrium conditions via Newton-Raphson starting from initial_roots."""
    x = np.array(initial_roots, dtype=np.float64)
    n = len(x)
    for it in range(max_iter):
        res = _equilibrium_residuals(x, a)
        if np.max(np.abs(res)) < tol:
            break
        J = _equilibrium_jacobian(x, a)
        try:
            dx = np.linalg.solve(J, -res)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            dx = np.linalg.pinv(J) @ (-res)
        # Damping if step too large
        step_norm = np.linalg.norm(dx)
        if step_norm > 1.0:
            dx *= 1.0 / step_norm
        x_new = x + dx
        # Ensure positivity
        if np.any(x_new <= 0):
            # Try smaller step
            for frac in [0.5, 0.25, 0.1, 0.05, 0.01]:
                x_try = x + frac * dx
                if np.all(x_try > 0):
                    x_new = x_try
                    break
            else:
                break
        x = x_new
    return x, np.max(np.abs(_equilibrium_residuals(x, a)))

def _c4_from_alpha_gas(a, initial_roots):
    """Solve equilibrium for parameter a, compute C4."""
    try:
        roots, residual = _solve_equilibrium(a, initial_roots)
        if not np.all(np.isfinite(roots)) or not np.all(roots > 0):
            return np.inf
        if residual > 1e-6:
            return np.inf
        return _compute_c4(roots)
    except Exception:
        return np.inf

def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 60.0

    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    # Initial guess: standard Laguerre roots
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0).astype(np.float64)
    best_c4 = _compute_c4(x0)
    best_roots = x0.copy()

    # Determine what 'a' corresponds to the initial Laguerre configuration
    # At equilibrium: a/x_i = 2*sum_{j!=i} 1/(x_i-x_j)^3
    # So a = x_i * 2*sum_{j!=i} 1/(x_i-x_j)^3 (should be same for all i at equilibrium)
    # Laguerre roots are NOT at equilibrium of this gas, so we need to find a that works.
    # Let's just search over a directly.

    # Phase 1: 1D search over a (log-spaced)
    best_a = None
    best_c4_gas = np.inf

    # The parameter a controls the density/spread of roots.
    # For larger a, roots are pushed further apart (confinement stronger).
    # Let's search over a range.
    log_a_grid = np.linspace(math.log(0.01), math.log(10000.0), 200)
    for la in log_a_grid:
        if not time_ok():
            break
        a_val = np.exp(la)
        # Use current best roots as initial guess, or spread them
        # Scale initial roots to give better starting point
        scale = (a_val / 12.0) ** 0.5  # rough scaling
        init = x0 * max(scale, 0.1)
        init = np.sort(init)
        c4_val = _c4_from_alpha_gas(a_val, init)
        if c4_val < best_c4_gas:
            best_c4_gas = c4_val
            best_a = a_val
            # Also try with unscaled initial
            c4_val2 = _c4_from_alpha_gas(a_val, x0)
            if c4_val2 < best_c4_gas:
                best_c4_gas = c4_val2
                best_a = a_val

    # Refine best a with scalar minimization
    if best_a is not None and time_ok():
        a_lo = max(best_a * 0.3, 0.001)
        a_hi = min(best_a * 3.0, 100000.0)
        try:
            res_a = minimize_scalar(
                lambda la: _c4_from_alpha_gas(np.exp(la), x0),
                bounds=(math.log(a_lo), math.log(a_hi)),
                method='bounded',
                options={'xatol': 1e-10, 'maxiter': 300}
            )
            if res_a.success and np.isfinite(res_a.fun) and res_a.fun < best_c4_gas:
                best_a = np.exp(res_a.x)
                best_c4_gas = res_a.fun
        except Exception:
            pass

    # Get roots from best gas parameter
    if best_a is not None:
        gas_roots, _ = _solve_equilibrium(best_a, x0)
        if np.all(np.isfinite(gas_roots)) and np.all(gas_roots > 0):
            gas_roots = np.sort(gas_roots)
            c4_gas = _compute_c4(gas_roots)
            if c4_gas < best_c4:
                best_c4 = c4_gas
                best_roots = gas_roots.copy()

    # Phase 2: Full 12-DOF L-BFGS-B refinement from gas equilibrium point
    if time_ok():
        def _objective(roots):
            return _compute_c4(roots)

        def _objective_grad(roots):
            roots = np.asarray(roots, dtype=np.float64)
            if np.any(roots <= 0) or np.any(np.isnan(roots)):
                return np.inf, np.zeros_like(roots)
            B = polyfromroots(roots)
            P = polymul(B, B)
            P2 = polymul(P, P)
            I1 = _poly_moment(P)
            I2 = _poly_moment(P2)
            if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
                return np.inf, np.zeros_like(roots)
            c4 = I2 / (I1 * I1)
            # Analytic gradient
            n = len(roots)
            B3 = polymul(polymul(B, B), B)
            grad = np.zeros(n)
            for i in range(n):
                R_i = np.array([1.0])
                for j in range(n):
                    if j == i:
                        continue
                    R_i = polymul(R_i, np.array([-roots[j], 1.0]))
                BR_i = polymul(B, R_i)
                dI1_i = -2.0 * _poly_moment(BR_i)
                B3R_i = polymul(B3, R_i)
                dI2_i = -4.0 * _poly_moment(B3R_i)
                grad[i] = dI2_i / (I1 * I1) - 2.0 * I2 * dI1_i / (I1 ** 3)
            return float(c4), grad

        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-8, None)] * 12,
                options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 3: Multi-start with different gas parameters
    if time_ok():
        for a_try in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
            if not time_ok():
                break
            init_roots, _ = _solve_equilibrium(a_try, x0)
            if not (np.all(np.isfinite(init_roots)) and np.all(init_roots > 0)):
                continue
            init_roots = np.sort(init_roots)
            try:
                res = minimize(
                    _objective_grad,
                    init_roots,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(1e-8, None)] * 12,
                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 40}
                )
                if np.all(res.x > 0) and np.isfinite(res.fun):
                    candidate = np.sort(res.x)
                    c4 = _compute_c4(candidate)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
            except Exception:
                pass

    # Phase 4: Differential evolution for global search
    if time_ok():
        from scipy.optimize import differential_evolution
        bounds = [(1e-4, 200.0)] * 12
        try:
            de_result = differential_evolution(
                _objective,
                bounds=bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-10,
                polish=False,
                mutation=(0.4, 1.0),
                recombination=0.7,
                update_interval=15
            )
            if np.all(de_result.x > 0) and np.isfinite(de_result.fun):
                candidate = np.sort(de_result.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 5: Final polish
    for polish_iter in range(4):
        if not time_ok():
            break
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-10, None)] * 12,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 30}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
                else:
                    break
        except Exception:
            break

    # Final safety check
    if not np.all(np.isfinite(best_roots)) or not np.all(best_roots > 0):
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)

    return [float(xi) for xi in best_roots]
