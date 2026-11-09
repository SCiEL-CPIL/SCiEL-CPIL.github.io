"""Growth-index optimization: minimize worst-case relative error of f and D."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# ---------------- Reference GR solution ----------------

def _Om(a, Om0, w0, wa):
    """Matter density fraction Om(a) for CPL dark energy."""
    a = np.asarray(a, dtype=float)
    # E^2 = Om0 a^-3 + (1-Om0) a^(-3(1+w0+wa)) * a^(3wa)
    # a^(-3(1+w0+wa)) * a^(3wa) = a^(-3 - 3w0)
    # So dark energy term = (1-Om0) * a^(-3-3w0) * a^(3wa) = (1-Om0) * a^(-3(1+w0) + 3wa)
    # Wait, let me redo. rho_de ∝ a^{-3(1+w)} where w = w0 + wa(1-a)
    # ln(rho_de) = -3 ∫_1^a (1+w(a'))/a' da'
    # = -3 ∫_1^a (1+w0+wa(1-a'))/a' da'
    # = -3[(1+w0)∫_1^a da'/a' + wa∫_1^a (1-a')/a' da']
    # = -3[(1+w0)ln a + wa(ln a - (a-1))]
    # So rho_de ∝ a^{-3(1+w0)} * exp(-3wa(ln a - a + 1))
    # = a^{-3(1+w0)} * a^{-3wa} * exp(3wa(a-1))
    # = a^{-3(1+w0+wa)} * exp(3wa(a-1))
    # E^2 = Om0 a^-3 + (1-Om0) a^{-3(1+w0+wa)} exp(3wa(a-1))
    term_m = Om0 * a**(-3)
    term_de = (1.0 - Om0) * a**(-3.0*(1.0 + w0 + wa)) * np.exp(3.0 * wa * (a - 1.0))
    return term_m / (term_m + term_de)

def _gr_ode(a, y, Om0, w0, wa):
    """Second-order ODE for growth: a^2 D'' + [3 - (1/2)Om(a) + (3/2)w(a)] a D' - (3/2)Om(a) D = 0
    In terms of x = ln a: D''_x + [3 - (1/2)Om + (3/2)w] D'_x - (3/2)Om D = 0
    where w = w0 + wa(1-a).
    State: [D, D'_x]
    """
    a_val = math.exp(a)
    om = _Om(a_val, Om0, w0, wa)
    w = w0 + wa * (1.0 - a_val)
    d1 = y[1]
    d2 = -(3.0 - 0.5 * om + 1.5 * w) * d1 + 1.5 * om * y[0]
    return [d1, d2]

def _gr_reference(a_grid, Om0, w0, wa):
    """Solve GR ODE and return (f_ref, D_norm_ref) on a_grid.
    f_ref = D'_x / D (the logarithmic derivative)
    D_norm_ref = D(a)/D(1)
    """
    a_min = a_grid[0]
    a_max = a_grid[-1]
    x_min = math.log(a_min)
    x_max = math.log(a_max)

    # Initial conditions at x=0 (a=1): D(1)=1, D'_x(1) = f(1) = Om0^gamma... 
    # Actually we need the true GR initial condition. At a=1, D=1, D'_x = f_GR(1).
    # f_GR(1) is determined by the ODE; we can integrate backwards from a=1 to a_min
    # and forwards to a_max, or just integrate over [x_min, x_max] with ICs at x=0.
    # Better: integrate from x=0 outward in both directions, or use a single integration
    # with ICs at x=0 and dense output.

    # Use solve_ivp from x_min to x_max with ICs at x=0? That's not directly supported.
    # Instead, integrate from x_min to x_max, but we need ICs. 
    # Standard approach: integrate from a=1 (x=0) to a=a_min (backwards) and to a=a_max (forwards).
    # Or just integrate over the full range with ICs at x=0 using a trick.
    # Simplest: integrate from x_min to x_max, setting ICs at x=0 by splitting.

    # Let's integrate from x_min to x_max with initial conditions at x_min.
    # We need D(x_min) and D'(x_min). We can get these by integrating from x=0 backwards.
    # For simplicity and robustness, integrate from x=0 to x_max (forward) and from x=0 to x_min (backward).

    # Forward: x=0 to x_max
    x_fwd = np.linspace(0.0, x_max, 200)
    sol_fwd = solve_ivp(_gr_ode, [0.0, x_max], [1.0, 0.55], args=(Om0, w0, wa),
                        t_eval=x_fwd, rtol=1e-10, atol=1e-12, method='RK45')
    # Backward: x=0 to x_min
    x_bwd = np.linspace(0.0, x_min, 200)
    sol_bwd = solve_ivp(_gr_ode, [0.0, x_min], [1.0, 0.55], args=(Om0, w0, wa),
                        t_eval=x_bwd, rtol=1e-10, atol=1e-12, method='RK45')

    # Build reference on a_grid
    D_vals = np.zeros_like(a_grid)
    Dp_vals = np.zeros_like(a_grid)
    for i, a_val in enumerate(a_grid):
        x_val = math.log(a_val)
        if x_val >= 0:
            idx = np.argmin(np.abs(x_fwd - x_val))
            D_vals[i] = sol_fwd.y[0, idx]
            Dp_vals[i] = sol_fwd.y[1, idx]
        else:
            idx = np.argmin(np.abs(x_bwd - x_val))
            D_vals[i] = sol_bwd.y[0, idx]
            Dp_vals[i] = sol_bwd.y[1, idx]

    f_ref = Dp_vals / D_vals
    D_norm = D_vals / D_vals[np.argmin(np.abs(a_grid - 1.0))]
    return f_ref, D_norm

# ---------------- Approximation functions ----------------

def _gamma_approx(coeffs, a, Om0, w0, wa):
    """Compute gamma(a) = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q"""
    c0, c1, c2, c3, c4, c5 = coeffs
    om = _Om(a, Om0, w0, wa)
    q = 1.0 - om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q

def _f_approx(coeffs, a, Om0, w0, wa):
    """f(a) = Om(a)^gamma(a)"""
    om = _Om(a, Om0, w0, wa)
    g = _gamma_approx(coeffs, a, Om0, w0, wa)
    return om**g

def _D_norm_approx(coeffs, a_grid, Om0, w0, wa):
    """D(a)/D(1) = exp(-∫_0^{ln a} f(exp(x)) dx)"""
    n = len(a_grid)
    D_norm = np.zeros(n)
    D_norm[0] = 1.0
    # Integrate from a=1 to each a
    # Find index of a=1
    idx_one = np.argmin(np.abs(a_grid - 1.0))
    for i in range(n):
        if i == idx_one:
            D_norm[i] = 1.0
            continue
        if i > idx_one:
            # Integrate from a_grid[idx_one] to a_grid[i]
            xs = np.linspace(math.log(a_grid[idx_one]), math.log(a_grid[i]), 200)
            f_vals = _f_approx(coeffs, np.exp(xs), Om0, w0, wa)
            integral = np.trapz(f_vals, xs)
            D_norm[i] = math.exp(-integral)
        else:
            # Integrate from a_grid[i] to a_grid[idx_one]
            xs = np.linspace(math.log(a_grid[i]), math.log(a_grid[idx_one]), 200)
            f_vals = _f_approx(coeffs, np.exp(xs), Om0, w0, wa)
            integral = np.trapz(f_vals, xs)
            D_norm[i] = math.exp(integral)
    return D_norm

# ---------------- Objective function ----------------

def _objective(coeffs, a_grid, f_refs, D_refs, Om0s, w0s, was, deadline):
    """Compute smooth max relative error of f and D_norm."""
    if time.time() > deadline:
        return 1e10
    total_err = 0.0
    max_val = 0.0
    for k in range(len(Om0s)):
        Om0, w0, wa = Om0s[k], w0s[k], was[k]
        f_app = _f_approx(coeffs, a_grid, Om0, w0, wa)
        D_app = _D_norm_approx(coeffs, a_grid, Om0, w0, wa)
        f_ref = f_refs[k]
        D_ref = D_refs[k]
        # Relative errors
        err_f = np.abs(f_app - f_ref) / np.maximum(np.abs(f_ref), 1e-12)
        err_D = np.abs(D_app - D_ref) / np.maximum(np.abs(D_ref), 1e-12)
        all_err = np.concatenate([err_f, err_D])
        total_err += np.sum(all_err)
        max_val = max(max_val, np.max(all_err))
    # Smooth max via log-sum-exp
    if total_err > 0:
        smooth_max = math.log(total_err)
    else:
        smooth_max = max_val
    # Gamma constraint penalty
    for k in range(len(Om0s)):
        Om0, w0, wa = Om0s[k], w0s[k], was[k]
        g = _gamma_approx(coeffs, a_grid, Om0, w0, wa)
        penalty = 0.0
        penalty += np.sum(np.maximum(0.0, 0.1 - g)**2)
        penalty += np.sum(np.maximum(0.0, g - 1.2)**2)
        smooth_max += 10.0 * penalty
    return smooth_max

# ---------------- Main construct function ----------------

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin

    # Validation cosmologies (CPL dark energy models)
    # Om0 in [0.2, 0.4], w0 in [-1.2, 0.0], wa in [-0.5, 0.5]
    Om0s = [0.25, 0.30, 0.35, 0.30, 0.25, 0.35, 0.30, 0.27, 0.32, 0.30]
    w0s  = [-1.0, -0.8, -1.2, -0.5, 0.0, -1.0, -0.9, -0.7, -1.1, -1.0]
    was  = [0.0, 0.3, -0.5, 0.2, -0.3, 0.1, -0.2, 0.4, 0.0, -0.1]

    # a grid: log-spaced from 0.05 to 1.0
    a_grid = np.geomspace(0.05, 1.0, 50)

    # Precompute GR references
    f_refs = []
    D_refs = []
    for k in range(len(Om0s)):
        f_ref, D_ref = _gr_reference(a_grid, Om0s[k], w0s[k], was[k])
        f_refs.append(f_ref)
        D_refs.append(D_ref)

    # Initial guess: constant gamma = 0.55
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    best_x = x0.copy()
    best_obj = _objective(x0, a_grid, f_refs, D_refs, Om0s, w0s, was, deadline)

    # Optimize with Nelder-Mead
    if time.time() < deadline - 10.0:
        try:
            res = minimize(_objective, x0, args=(a_grid, f_refs, D_refs, Om0s, w0s, was, deadline),
                           method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
        except Exception:
            pass

    # Final polish with tighter tolerance
    if time.time() < deadline - 5.0:
        try:
            res2 = minimize(_objective, best_x, args=(a_grid, f_refs, D_refs, Om0s, w0s, was, deadline),
                            method='Nelder-Mead',
                            options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
            if res2.fun < best_obj:
                best_obj = res2.fun
                best_x = res2.x
        except Exception:
            pass

    # Clip to [-10, 10] and ensure finite
    result = np.clip(best_x, -10.0, 10.0)
    result = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
    return [float(v) for v in result]
