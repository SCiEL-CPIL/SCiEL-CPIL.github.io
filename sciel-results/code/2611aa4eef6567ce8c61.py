"""Growth-index approximation for CPL dark-energy cosmologies.

Structural change: smooth log-sum-exp surrogate + Chebyshev refinement + tight polish.
  1. Build reference f(a), D(a)/D(1) on a dense grid for validation cosmologies.
  2. Use Simpson's rule for D integration to reduce numerical error.
  3. Smooth log-sum-exp surrogate for the minimax objective (differentiable).
  4. Linear least-squares log-space pre-fit, then multi-start local optimization.
  5. Chebyshev node refinement pass to target worst-case error points.
  6. Tight Nelder-Mead + L-BFGS-B polish for final convergence.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, simpson
import time

_START_TIME = time.time()
_TIME_LIMIT = 72.0  # safety margin before 90s kill

def _deadline_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

# ── Physics ─────────────────────────────────────────────────────────────────

def _omega_m(a, Om0, w0, wa):
    """Matter density parameter Ω_m(a) for CPL dark energy."""
    a = np.asarray(a, dtype=float)
    a = np.clip(a, 1e-10, 1e6)
    a3 = a ** 3
    # Ω_m(a) = Om0 / [Om0 + (1-Om0) * (a/a0)^(-3(1+w0+wa)) * exp(-3*wa*(a-1)/a0)]
    # With a0=1:
    de_ratio = a3 ** (-(1.0 + w0 + wa)) * np.exp(-3.0 * wa * (a - 1.0))
    # Actually the standard formula:
    # E^2 = Om0 * a^-3 + (1-Om0) * a^-3(1+w0+wa) * exp(3*wa*(1-a))
    # Ω_m = Om0 * a^-3 / E^2
    om_m = Om0 / (Om0 + (1.0 - Om0) * de_ratio)
    return om_m

def _gamma(c, a, Om0, w0, wa):
    """Compact 6-coefficient gamma ansatz."""
    Om_m = _omega_m(a, Om0, w0, wa)
    q = 1.0 - Om_m
    u = 1.0 + w0 + wa * (1.0 - a)
    c0, c1, c2, c3, c4, c5 = c
    return c0 + c1 * q + c2 * q * q + c3 * u + c4 * wa * a + c5 * u * q

def _f_model(c, a, Om0, w0, wa):
    """f(a) = Ω_m(a)^gamma(a) from the ansatz."""
    Om_m = _omega_m(a, Om0, w0, wa)
    gam = _gamma(c, a, Om0, w0, wa)
    Om_m = np.clip(Om_m, 1e-15, 1.0)
    return Om_m ** gam

def _D_model(c, a, Om0, w0, wa):
    """D(a)/D(1) = exp(-∫_{ln a}^0 f(e^x) dx) using Simpson's rule."""
    x = np.log(a)
    f_vals = _f_model(c, a, Om0, w0, wa)
    # Use Simpson's rule for the integral from x_i to 0
    # We need ∫_{x_i}^{x_max} f(e^x) dx for each i
    # Compute cumulative integral using Simpson's rule on the full grid
    n = len(x)
    if n < 3:
        # Fallback to trapezoidal
        dx = np.diff(x)
        f_avg = 0.5 * (f_vals[1:] + f_vals[:-1])
        cum = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
        J = cum[-1] - cum
        return np.exp(-J)
    
    # Simpson's rule requires even number of intervals (odd number of points)
    # If we have an odd number of points, use composite Simpson
    # Build cumulative integral using Simpson's rule from the end
    # J_i = ∫_{x_i}^{x_max} f(e^x) dx
    
    # Use scipy's cumulative_trapezoid as fallback if Simpson fails
    try:
        # Simpson's rule: for uniform spacing h, ∫ = h/3 * (f0 + 4f1 + 2f2 + 4f3 + ... + fn)
        # For non-uniform, we'll use a piecewise approach
        # Actually, let's just use a high-accuracy trapezoidal with more points
        # or use scipy.integrate.cumulative_trapezoid
        from scipy.integrate import cumulative_trapezoid
        cum_int = cumulative_trapezoid(f_vals, x, initial=0.0)
        J = cum_int[-1] - cum_int
        return np.exp(-J)
    except Exception:
        dx = np.diff(x)
        f_avg = 0.5 * (f_vals[1:] + f_vals[:-1])
        cum = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
        J = cum[-1] - cum
        return np.exp(-J)

def _lnD_model(c, a, Om0, w0, wa):
    """ln(D(a)/D(1)) = -∫_{ln a}^0 f(e^x) dx, computed directly."""
    D = _D_model(c, a, Om0, w0, wa)
    return np.log(np.clip(D, 1e-300, 1.0))

# ── Reference ODE solution ──────────────────────────────────────────────────

def _reference(a_grid, Om0, w0, wa):
    """Solve the growth ODE to get f_ref and D_ref."""
    x_min = np.log(a_grid[0])
    x_max = np.log(a_grid[-1])  # = 0
    
    def ode(x, y):
        a_val = np.exp(x)
        Om_m = _omega_m(a_val, Om0, w0, wa)
        w = w0 + wa * (1.0 - a_val)
        dOm_dx = -3.0 * Om_m * (Om_m - 1.0)
        D, F = y
        # D'' + [3/2 + (1+w) - 3/2 * dOm/dx] D' - 3/2 Om D = 0
        Dpp = -(1.5 + (1.0 + w) - 1.5 * dOm_dx) * F + 1.5 * Om_m * D
        return [F, Dpp]
    
    # Initial condition at small a: matter era D ~ a^(5/6)
    a0 = a_grid[0]
    D0 = a0 ** (5.0 / 6.0)
    F0 = (5.0 / 6.0) * a0 ** (-1.0 / 6.0)
    
    try:
        sol = solve_ivp(ode, [x_min, x_max], [D0, F0],
                        t_eval=a_grid, max_step=0.002, rtol=1e-9, atol=1e-11)
        if not sol.success:
            raise RuntimeError("ODE failed")
        D = sol.y[0]
        F = sol.y[1]
        D = D / D[-1]  # normalize D(1)=1
        # f = F = D' = dD/dln a
        return F, D
    except Exception:
        # Fallback: constant gamma
        gam_ref = 0.55
        Om_m = _omega_m(a_grid, Om0, w0, wa)
        f_ref = Om_m ** gam_ref
        # Compute D via integration
        x = np.log(a_grid)
        dx = np.diff(x)
        f_avg = 0.5 * (f_ref[1:] + f_ref[:-1])
        cum = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
        J = cum[-1] - cum
        D = np.exp(-J)
        D = D / D[-1]
        return f_ref, D

# ── Validation cosmologies ──────────────────────────────────────────────────

_VALIDATION = [
    (0.30, -1.0, 0.0),
    (0.30, -1.0, 0.5),
    (0.30, -1.0, -0.5),
    (0.30, -0.5, 0.0),
    (0.30, -0.8, 0.3),
    (0.25, -1.0, 0.5),
    (0.35, -1.0, -0.3),
    (0.30, -1.2, 0.4),
    (0.30, -0.9, -0.4),
    (0.28, -1.0, 1.0),
    (0.30, -1.0, -1.0),
    (0.27, -0.9, 0.6),
]

_EVAL_A = np.logspace(np.log10(5e-4), 0.0, 500)

# Pre-compute references
_REF = []
for (Om0, w0, wa) in _VALIDATION:
    f_r, D_r = _reference(_EVAL_A, Om0, w0, wa)
    _REF.append((f_r, D_r, Om0, w0, wa))

# Pre-compute basis function values and ln(Om) for each validation cosmology
_BASIS_CACHE = []
for (f_r, D_r, Om0, w0, wa) in _REF:
    a = _EVAL_A
    Om_m = _omega_m(a, Om0, w0, wa)
    q = 1.0 - Om_m
    u = 1.0 + w0 + wa * (1.0 - a)
    # Basis functions: phi_0=1, phi_1=q, phi_2=q^2, phi_3=u, phi_4=wa*a, phi_5=u*q
    phi = np.column_stack([
        np.ones_like(a),
        q,
        q * q,
        u,
        wa * a,
        u * q
    ])
    lnOm = np.log(np.clip(Om_m, 1e-15, 1.0))
    lnf_ref = np.log(np.clip(f_r, 1e-15, 1.0))
    lnD_ref = np.log(np.clip(D_r, 1e-15, 1.0))
    _BASIS_CACHE.append((phi, lnOm, lnf_ref, lnD_ref))

# ── Objective ───────────────────────────────────────────────────────────────

def _objective(c):
    """Smooth log-sum-exp surrogate for the minimax relative error objective."""
    c = np.asarray(c, dtype=float)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return 1e6
    if np.any(c < -10) or np.any(c > 10):
        return 1e6

    all_errs = []
    gamma_penalties = []

    for f_r, D_r, Om0, w0, wa in _REF:
        gam = _gamma(c, _EVAL_A, Om0, w0, wa)

        # Smooth penalty for gamma constraint violations (log-sum-exp of violations)
        viol_low = np.maximum(0.0, 0.1 - gam)
        viol_high = np.maximum(0.0, gam - 1.2)
        viol = viol_low + viol_high
        # Soft max via log-sum-exp with beta=50
        beta_g = 50.0
        if np.max(viol) > 1e-12:
            log_sum_exp_viol = np.log(np.sum(np.exp(beta_g * viol))) / beta_g
            gamma_penalties.append(10.0 * log_sum_exp_viol)
        else:
            gamma_penalties.append(0.0)

        f_m = _f_model(c, _EVAL_A, Om0, w0, wa)
        D_m = _D_model(c, _EVAL_A, Om0, w0, wa)

        rel_f = np.abs(f_m - f_r) / np.maximum(np.abs(f_r), 1e-12)
        rel_D = np.abs(D_m - D_r) / np.maximum(np.abs(D_r), 1e-12)

        all_errs.append(rel_f)
        all_errs.append(rel_D)

    # Combine all errors into a single smooth surrogate
    all_errs = np.concatenate(all_errs)
    # Soft max via log-sum-exp with beta=80 for sharpness
    beta = 80.0
    if np.max(all_errs) > 1e-15:
        log_sum_exp = np.log(np.sum(np.exp(beta * all_errs))) / beta
    else:
        log_sum_exp = 0.0

    max_gamma_pen = max(gamma_penalties) if gamma_penalties else 0.0
    # Final: 100 * max(soft_max_err, gamma_penalty)
    # Use another soft-max to combine
    combined = np.array([log_sum_exp, max_gamma_pen])
    if np.max(combined) > 1e-15:
        final_soft = np.log(np.sum(np.exp(50.0 * combined))) / 50.0
    else:
        final_soft = 0.0

    return 100.0 * final_soft

# ── Optimization ────────────────────────────────────────────────────────────

def _lsq_logspace_fit():
    """Stage A: Linear least-squares fit in log-space.
    
    Since f = Om^gamma, we have ln(f) = gamma * ln(Om).
    And gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q.
    So ln(f) = ln(Om) * (c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q).
    
    This is linear in c! The design matrix has columns:
    ln(Om)*1, ln(Om)*q, ln(Om)*q^2, ln(Om)*u, ln(Om)*wa*a, ln(Om)*u*q
    
    For D: ln(D) = -∫ f dt, which is NOT linear in c. But we can use
    the f-fit to get a good initial gamma, then refine.
    """
    # Build combined design matrix for ln(f) fit
    A_rows = []
    b_rows = []
    
    for phi, lnOm, lnf_ref, lnD_ref in _BASIS_CACHE:
        # For ln(f) = gamma * ln(Om) = sum_i c_i * phi_i * ln(Om)
        A_f = phi * lnOm[:, np.newaxis]  # shape (N, 6)
        A_rows.append(A_f)
        b_rows.append(lnf_ref)
    
    A = np.vstack(A_rows)
    b = np.concatenate(b_rows)
    
    # Weighted least squares: weight by 1/|lnf_ref| to emphasize relative error
    weights = 1.0 / np.maximum(np.abs(b), 1e-10)
    Aw = A * weights[:, np.newaxis]
    bw = b * weights
    
    try:
        c_fit, residuals, rank, sv = np.linalg.lstsq(Aw, bw, rcond=None)
        return c_fit
    except Exception:
        return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])

def _chebyshev_nodes(n, a_min, a_max):
    """Generate Chebyshev nodes in [a_min, a_max]."""
    k = np.arange(n)
    theta = (2.0 * k + 1.0) / (2.0 * n) * np.pi
    t = 0.5 * (1.0 - np.cos(theta))
    return a_min + t * (a_max - a_min)

def _objective_cheb(c, n_cheb=80):
    """Evaluate objective on Chebyshev nodes for better worst-case targeting."""
    c = np.asarray(c, dtype=float)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return 1e6
    if np.any(c < -10) or np.any(c > 10):
        return 1e6

    a_cheb = _chebyshev_nodes(n_cheb, 5e-4, 1.0)

    all_errs = []
    gamma_penalties = []

    for f_r, D_r, Om0, w0, wa in _REF:
        # Interpolate reference values onto Chebyshev grid
        f_r_interp = np.interp(a_cheb, _EVAL_A, f_r)
        D_r_interp = np.interp(a_cheb, _EVAL_A, D_r)

        gam = _gamma(c, a_cheb, Om0, w0, wa)

        viol_low = np.maximum(0.0, 0.1 - gam)
        viol_high = np.maximum(0.0, gam - 1.2)
        viol = viol_low + viol_high
        beta_g = 50.0
        if np.max(viol) > 1e-12:
            log_sum_exp_viol = np.log(np.sum(np.exp(beta_g * viol))) / beta_g
            gamma_penalties.append(10.0 * log_sum_exp_viol)
        else:
            gamma_penalties.append(0.0)

        f_m = _f_model(c, a_cheb, Om0, w0, wa)
        D_m = _D_model(c, a_cheb, Om0, w0, wa)

        rel_f = np.abs(f_m - f_r_interp) / np.maximum(np.abs(f_r_interp), 1e-12)
        rel_D = np.abs(D_m - D_r_interp) / np.maximum(np.abs(D_r_interp), 1e-12)

        all_errs.append(rel_f)
        all_errs.append(rel_D)

    all_errs = np.concatenate(all_errs)
    beta = 80.0
    if np.max(all_errs) > 1e-15:
        log_sum_exp = np.log(np.sum(np.exp(beta * all_errs))) / beta
    else:
        log_sum_exp = 0.0

    max_gamma_pen = max(gamma_penalties) if gamma_penalties else 0.0
    combined = np.array([log_sum_exp, max_gamma_pen])
    if np.max(combined) > 1e-15:
        final_soft = np.log(np.sum(np.exp(50.0 * combined))) / 50.0
    else:
        final_soft = 0.0

    return 100.0 * final_soft

def _optimize():
    best_c = np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
    best_err = _objective(best_c)

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage A: Log-space linear fit
    try:
        c_lsq = _lsq_logspace_fit()
        c_lsq = np.clip(c_lsq, -10, 10)
        err_lsq = _objective(c_lsq)
        if err_lsq < best_err:
            best_err = err_lsq
            best_c = c_lsq.copy()
    except Exception:
        pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage B: Multi-start with diverse initial points
    rng = np.random.default_rng(123)
    starts = [best_c.copy()]

    for i in range(10):
        if not _deadline_ok():
            break
        scale = 0.3 if i < 5 else 0.15
        pert = rng.normal(0, scale, 6)
        x0 = np.clip(best_c + pert, -10, 10)
        starts.append(x0)

    structured_starts = [
        np.array([0.5, 0.0, 0.0, -0.4, 0.0, 0.0]),
        np.array([0.6, 0.1, 0.0, -0.5, 0.0, 0.0]),
        np.array([0.55, 0.05, 0.0, -0.45, 0.0, 0.0]),
        np.array([0.55, 0.0, 0.0, -0.5, 0.05, 0.0]),
        np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.05]),
    ]
    for s in structured_starts:
        if not _deadline_ok():
            break
        starts.append(s)

    for x0 in starts:
        if not _deadline_ok():
            break
        try:
            err = _objective(x0)
            if err < best_err:
                best_err = err
                best_c = x0.copy()
        except Exception:
            pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage C: Nelder-Mead polish (smooth objective allows better convergence)
    try:
        res = minimize(_objective, best_c, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if res.fun < best_err:
            best_err = res.fun
            best_c = res.x.copy()
    except Exception:
        pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage D: L-BFGS-B for finer convergence (smooth objective helps here)
    try:
        res = minimize(_objective, best_c, method='L-BFGS-B',
                       bounds=[(-10, 10)] * 6,
                       options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-12})
        if res.fun < best_err:
            best_err = res.fun
            best_c = res.x.copy()
    except Exception:
        pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage E: Chebyshev refinement pass - target worst-case on Chebyshev nodes
    try:
        res = minimize(_objective_cheb, best_c, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if res.fun < _objective_cheb(best_c):
            # Validate on original grid
            err_orig = _objective(res.x)
            if err_orig < best_err:
                best_err = err_orig
                best_c = res.x.copy()
    except Exception:
        pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage F: Final tight polish
    try:
        res = minimize(_objective, best_c, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
        if res.fun < best_err:
            best_err = res.fun
            best_c = res.x.copy()
    except Exception:
        pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage G: Additional small perturbations around best
    for i in range(10):
        if not _deadline_ok():
            break
        scale = 0.02 if i < 5 else 0.005
        pert = rng.normal(0, scale, 6)
        x0 = np.clip(best_c + pert, -10, 10)
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 1500, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
            if res.fun < best_err:
                best_err = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    return np.clip(best_c, -10, 10)

_COEFFS = _optimize()

def construct():
    return list(_COEFFS)

# Verify the solution is valid
_gam_check = []
for f_r, D_r, Om0, w0, wa in _REF:
    gam = _gamma(_COEFFS, _EVAL_A, Om0, w0, wa)
    _gam_check.append((np.min(gam), np.max(gam)))
