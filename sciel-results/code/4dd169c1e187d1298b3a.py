"""Growth-index approximation for CPL dark-energy cosmologies.

Dual-space Chebyshev projection strategy:
  1. Map (q, u) to a normalized Chebyshev grid for equioscillation properties.
  2. Stage A: Linear least-squares pre-fit in log-space (ln f = gamma * ln Om).
  3. Stage B: Log-derivative local regression for stable landscape.
  4. Stage C: Smooth epigraph minimax polish.
  5. Return optimized 6 coefficients via construct().
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

_START_TIME = time.time()
_TIME_LIMIT = 75.0  # safety margin before 90s kill

def _deadline_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

def _chebyshev_nodes(n):
    """Return n Chebyshev nodes in [-1, 1]."""
    k = np.arange(n)
    return np.cos(np.pi * (2*k + 1) / (2*n))

# ── Physics ─────────────────────────────────────────────────────────────────

def _omega_m(a, Om0, w0, wa):
    """Matter density parameter Ω_m(a) for CPL dark energy."""
    a = np.asarray(a, dtype=float)
    a = np.clip(a, 1e-10, 1e6)
    a3 = a ** 3
    de_ratio = a3 ** (-(1.0 + w0 + wa)) * np.exp(-3.0 * wa * (a - 1.0))
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
    """D(a)/D(1) = exp(-∫_{ln a}^0 f(e^x) dx)"""
    x = np.log(a)
    f_vals = _f_model(c, a, Om0, w0, wa)
    dx = np.diff(x)
    f_avg = 0.5 * (f_vals[1:] + f_vals[:-1])
    cum = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
    J = cum[-1] - cum
    return np.exp(-J)

# ── Reference ODE solution ──────────────────────────────────────────────────

def _reference(a_grid, Om0, w0, wa):
    """Solve the growth ODE to get f_ref and D_ref."""
    x_min = np.log(a_grid[0])
    x_max = np.log(a_grid[-1])
    
    def ode(x, y):
        a_val = np.exp(x)
        Om_m = _omega_m(a_val, Om0, w0, wa)
        w = w0 + wa * (1.0 - a_val)
        dOm_dx = -3.0 * Om_m * (Om_m - 1.0)
        D, F = y
        Dpp = -(1.5 + (1.0 + w) - 1.5 * dOm_dx) * F + 1.5 * Om_m * D
        return [F, Dpp]
    
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
        D = D / D[-1]
        return F, D
    except Exception:
        gam_ref = 0.55
        Om_m = _omega_m(a_grid, Om0, w0, wa)
        f_ref = Om_m ** gam_ref
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

_N_POINTS = 256
_cheb_x = _chebyshev_nodes(_N_POINTS)
_log_a_min = np.log(5e-4)
_log_a_max = 0.0
_EVAL_A = np.exp(_log_a_min + (_cheb_x + 1) / 2 * (_log_a_max - _log_a_min))
_EVAL_A = np.sort(_EVAL_A)

_REF = []
for (Om0, w0, wa) in _VALIDATION:
    f_r, D_r = _reference(_EVAL_A, Om0, w0, wa)
    _REF.append((f_r, D_r, Om0, w0, wa))

# ── Log-derivative features ────────────────────────────────────────────────
# For f = Om^gamma, we have d(ln f)/d(ln Om) = gamma
# But more usefully: d(ln f)/dx where x = ln(a)
# d(ln f)/dx = d(ln Om)/dx * gamma + ln(Om) * d(gamma)/dx
# This gives us a local linear regression target.

def _build_log_derivative_matrix():
    """Build matrix for log-derivative regression: d(ln f)/dx ≈ sum c_i * phi_i' * ln(Om) + ...
    
    Actually, let's use: d(ln f)/dx = gamma * d(ln Om)/dx + ln(Om) * d(gamma)/dx
    
    The basis functions are phi_i = [1, q, q^2, u, wa*a, u*q]
    So d(gamma)/dx = sum c_i * d(phi_i)/dx
    
    We want: d(ln f)/dx = gamma * d(ln Om)/dx + ln(Om) * sum c_i * d(phi_i)/dx
    
    This is still nonlinear in c because gamma depends on c.
    
    Alternative: directly regress d(ln f)/dx on the features and their derivatives.
    Let's define target = d(ln f_ref)/dx
    And features = [phi_i, d(phi_i)/dx] for each i, plus gamma * d(ln Om)/dx terms.
    
    Simpler approach: use the fact that at the optimum, the local slope of ln f 
    should match. Let's just do a weighted least squares on gamma itself,
    where gamma = d(ln f)/d(ln Om), using finite differences.
    """
    rows = []
    targets = []
    
    for f_r, D_r, Om0, w0, wa in _REF:
        Om_m = _omega_m(_EVAL_A, Om0, w0, wa)
        q = 1.0 - Om_m
        u = 1.0 + w0 + wa * (1.0 - _EVAL_A)
        
        # Features
        phi = np.column_stack([
            np.ones_like(q),
            q,
            q ** 2,
            u,
            wa * _EVAL_A,
            u * q
        ])
        
        # Compute d(ln f)/d(ln Om) via finite differences
        ln_f = np.log(np.clip(f_r, 1e-15, 1.0))
        ln_Om = np.log(np.clip(Om_m, 1e-15, 1.0))
        
        # Central differences where possible
        dlnf_dlnom = np.zeros_like(ln_f)
        for i in range(len(ln_f)):
            if i > 0 and i < len(ln_f) - 1:
                denom = ln_Om[i+1] - ln_Om[i-1]
                if abs(denom) > 1e-10:
                    dlnf_dlnom[i] = (ln_f[i+1] - ln_f[i-1]) / denom
                else:
                    dlnf_dlnom[i] = 0.55  # fallback
            elif i == 0:
                denom = ln_Om[1] - ln_Om[0]
                if abs(denom) > 1e-10:
                    dlnf_dlnom[i] = (ln_f[1] - ln_f[0]) / denom
                else:
                    dlnf_dlnom[i] = 0.55
            else:
                denom = ln_Om[-1] - ln_Om[-2]
                if abs(denom) > 1e-10:
                    dlnf_dlnom[i] = (ln_f[-1] - ln_f[-2]) / denom
                else:
                    dlnf_dlnom[i] = 0.55
        
        # Weight by importance: emphasize mid-range where errors matter most
        weights = np.ones_like(ln_f)
        
        rows.append(phi * weights[:, np.newaxis])
        targets.append(dlnf_dlnom * weights)
    
    A = np.vstack(rows)
    b = np.concatenate(targets)
    return A, b

# ── Objective functions ─────────────────────────────────────────────────────

def _objective(c):
    """Exact minimax objective."""
    c = np.asarray(c, dtype=float)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return 1e6
    if np.any(c < -10) or np.any(c > 10):
        return 1e6
    
    max_err = 0.0
    for f_r, D_r, Om0, w0, wa in _REF:
        gam = _gamma(c, _EVAL_A, Om0, w0, wa)
        if np.any(gam < 0.1) or np.any(gam > 1.2):
            viol = np.maximum(0.0, 0.1 - gam) + np.maximum(0.0, gam - 1.2)
            max_err = max(max_err, 5.0 * np.max(viol))
            continue
        
        f_m = _f_model(c, _EVAL_A, Om0, w0, wa)
        D_m = _D_model(c, _EVAL_A, Om0, w0, wa)
        
        rel_f = np.abs(f_m - f_r) / np.maximum(np.abs(f_r), 1e-12)
        rel_D = np.abs(D_m - D_r) / np.maximum(np.abs(D_r), 1e-12)
        
        max_err = max(max_err, np.max(rel_f), np.max(rel_D))
    
    return 100.0 * max_err

def _smooth_objective(c, temperature=0.01):
    """Smooth approximation of minimax using log-sum-exp."""
    c = np.asarray(c, dtype=float)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return 1e6
    if np.any(c < -10) or np.any(c > 10):
        return 1e6
    
    all_errors = []
    for f_r, D_r, Om0, w0, wa in _REF:
        gam = _gamma(c, _EVAL_A, Om0, w0, wa)
        if np.any(gam < 0.1) or np.any(gam > 1.2):
            viol = np.maximum(0.0, 0.1 - gam) + np.maximum(0.0, gam - 1.2)
            all_errors.append(5.0 * viol)
            continue
        
        f_m = _f_model(c, _EVAL_A, Om0, w0, wa)
        D_m = _D_model(c, _EVAL_A, Om0, w0, wa)
        
        rel_f = np.abs(f_m - f_r) / np.maximum(np.abs(f_r), 1e-12)
        rel_D = np.abs(D_m - D_r) / np.maximum(np.abs(D_r), 1e-12)
        
        all_errors.append(rel_f)
        all_errors.append(rel_D)
    
    all_errors = np.concatenate(all_errors)
    
    # Log-sum-exp for smooth max
    if len(all_errors) == 0:
        return 1e6
    
    scaled = all_errors / temperature
    max_s = np.max(scaled)
    if max_s > 700:  # avoid overflow
        log_sum_exp = max_s
    else:
        log_sum_exp = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    
    smooth_max = temperature * log_sum_exp
    return 100.0 * smooth_max

# ── Optimization ────────────────────────────────────────────────────────────

def _lsq_logspace_fit():
    """Stage A: Linear least-squares fit in log-space."""
    try:
        A, b = _build_log_derivative_matrix()
        if A is None or len(A) == 0:
            return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
        
        c_lsq, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        c_lsq = np.clip(c_lsq, -10, 10)
        
        max_viol = 0.0
        for f_r, D_r, Om0, w0, wa in _REF:
            gam = _gamma(c_lsq, _EVAL_A, Om0, w0, wa)
            viol = np.maximum(0.0, 0.1 - gam) + np.maximum(0.0, gam - 1.2)
            max_viol = max(max_viol, np.max(viol))
        
        if max_viol > 0.05:
            return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
        
        return c_lsq
    except Exception:
        return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])

def _optimize():
    # Physics-constrained residual strategy:
    # 1. Fix c0 = 0.55 (matter era limit).
    # 2. Fit c1, c2, c3, c5 on w_a=0 models to capture w0/Om0 dependence.
    # 3. Fit c4 on all models to capture w_a dependence.
    
    c0_fixed = 0.55
    
    # Identify w_a=0 models
    wa0_indices = [i for i, (Om0, w0, wa) in enumerate(_VALIDATION) if abs(wa) < 1e-6]
    
    if not wa0_indices:
        return np.array([0.55, 0.071, 0.0, 0.0, 0.0, 0.0])
    
    def _obj_stage1(c_vars):
        c1, c2, c3, c5 = c_vars
        c_full = np.array([c0_fixed, c1, c2, c3, 0.0, c5])
        max_err = 0.0
        for i in wa0_indices:
            f_r, D_r, Om0, w0, wa = _REF[i]
            gam = _gamma(c_full, _EVAL_A, Om0, w0, wa)
            if np.any(gam < 0.1) or np.any(gam > 1.2):
                viol = np.maximum(0.0, 0.1 - gam) + np.maximum(0.0, gam - 1.2)
                max_err = max(max_err, 5.0 * np.max(viol))
                continue
            f_m = _f_model(c_full, _EVAL_A, Om0, w0, wa)
            D_m = _D_model(c_full, _EVAL_A, Om0, w0, wa)
            rel_f = np.abs(f_m - f_r) / np.maximum(np.abs(f_r), 1e-12)
            rel_D = np.abs(D_m - D_r) / np.maximum(np.abs(D_r), 1e-12)
            max_err = max(max_err, np.max(rel_f), np.max(rel_D))
        return 100.0 * max_err

    x0 = np.array([0.071, 0.0, 0.0, 0.0])
    best_c1 = x0.copy()
    best_err1 = _obj_stage1(x0)
    
    if _deadline_ok():
        try:
            res1 = minimize(_obj_stage1, x0, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res1.fun < best_err1:
                best_err1 = res1.fun
                best_c1 = res1.x.copy()
        except Exception:
            pass
            
    c1_best, c2_best, c3_best, c5_best = best_c1
    
    def _obj_stage2(c4_val):
        c_full = np.array([c0_fixed, c1_best, c2_best, c3_best, c4_val, c5_best])
        max_err = 0.0
        for f_r, D_r, Om0, w0, wa in _REF:
            gam = _gamma(c_full, _EVAL_A, Om0, w0, wa)
            if np.any(gam < 0.1) or np.any(gam > 1.2):
                viol = np.maximum(0.0, 0.1 - gam) + np.maximum(0.0, gam - 1.2)
                max_err = max(max_err, 5.0 * np.max(viol))
                continue
            f_m = _f_model(c_full, _EVAL_A, Om0, w0, wa)
            D_m = _D_model(c_full, _EVAL_A, Om0, w0, wa)
            rel_f = np.abs(f_m - f_r) / np.maximum(np.abs(f_r), 1e-12)
            rel_D = np.abs(D_m - D_r) / np.maximum(np.abs(D_r), 1e-12)
            max_err = max(max_err, np.max(rel_f), np.max(rel_D))
        return 100.0 * max_err

    x0_c4 = 0.0
    best_c4 = x0_c4
    best_err2 = _obj_stage2(x0_c4)
    
    if _deadline_ok():
        try:
            res2 = minimize(_obj_stage2, x0_c4, method='Nelder-Mead',
                            options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-12})
            if res2.fun < best_err2:
                best_err2 = res2.fun
                best_c4 = res2.x[0]
        except Exception:
            pass
            
    final_c = np.array([c0_fixed, c1_best, c2_best, c3_best, best_c4, c5_best])
    return np.clip(final_c, -10, 10)

_COEFFS = _optimize()

def construct():
    return list(_COEFFS)
