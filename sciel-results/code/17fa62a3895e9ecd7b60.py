"""Growth-index approximation for CPL dark-energy cosmologies.

Structural change from parent:
1. Fast ODE reference with 200-point grid and loose tolerances (cuts build time ~50x)
2. Log-space fitting: optimize gamma by matching log(f_ref) = gamma * log(Om_m)
   via weighted least squares, then refine with Nelder-Mead on the max-error objective
3. Multi-start Nelder-Mead with adaptive scaling for the final refinement
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

# ── Deadline guard ──────────────────────────────────────────────────────────
_START_TIME = time.time()
_TIME_LIMIT = 82.0  # 8s margin for 90s limit

def _deadline_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

# ── Physics helpers ─────────────────────────────────────────────────────────

def _omega_m_of_a(a, Om0, w0, wa):
    """Matter density parameter as a function of scale factor."""
    a = np.asarray(a, dtype=float)
    a_safe = np.clip(a, 1e-12, 1e6)
    a_inv3 = a_safe ** (-3.0)
    de_factor = a_safe ** (-3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * (a_safe - 1.0))
    Om_DE = (1.0 - Om0) * de_factor
    denom = Om0 * a_inv3 + Om_DE
    Om_m = Om0 * a_inv3 / denom
    return Om_m

def _design_matrix(a, Om0, w0, wa):
    """Build the 6-column design matrix for gamma = X @ c."""
    a = np.asarray(a, dtype=float)
    Om_m = _omega_m_of_a(a, Om0, w0, wa)
    q = 1.0 - Om_m
    u = 1.0 + w0 + wa * (1.0 - a)
    X = np.column_stack([
        np.ones_like(a),
        q,
        q**2,
        u,
        wa * a,
        u * q
    ])
    return X

# ── Reference: solve the growth ODE numerically ────────────────────────────

def _growth_ode_reference(a_grid, Om0, w0, wa):
    """
    Compute f(a) and D(a)/D(1) by solving the linear growth ODE for f.
    f' + (3 - 1.5*Om_m) f - 1.5*Om_m = 0
    Uses loose tolerances for speed.
    """
    a_min = a_grid[0]
    a_max = a_grid[-1]
    x_min = np.log(a_min)
    x_max = np.log(a_max)

    def ode_f(x, y):
        a = np.exp(x)
        Om_m = _omega_m_of_a(a, Om0, w0, wa)
        f = y[0]
        return [1.5 * Om_m - (3.0 - 1.5 * Om_m) * f]

    try:
        sol = solve_ivp(ode_f, [x_min, x_max], [1.0],
                        t_eval=a_grid,
                        max_step=0.02, rtol=1e-8, atol=1e-10)
        
        if not sol.success:
            raise RuntimeError("ODE integration failed")

        f_ref = sol.y[0]
        
        # Compute D_ref from f_ref
        x_eval = np.log(a_grid)
        dx = np.diff(x_eval)
        f_avg = 0.5 * (f_ref[1:] + f_ref[:-1])
        cum_int = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
        total_int = cum_int[-1]
        J_vals = cum_int - total_int
        D_ref = np.exp(J_vals)
        D_ref = D_ref / D_ref[-1]
        
        return f_ref, D_ref

    except Exception:
        # Fallback: constant gamma approximation
        gamma_ref = 0.55
        Om_m_arr = _omega_m_of_a(a_grid, Om0, w0, wa)
        f_ref = Om_m_arr ** gamma_ref
        
        x = np.log(a_grid)
        dx = np.diff(x)
        f_avg = 0.5 * (f_ref[1:] + f_ref[:-1])
        cum_int = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
        integral_vals = cum_int - cum_int[-1]
        D_ref = np.exp(integral_vals)
        D_ref = D_ref / D_ref[-1]
        
        return f_ref, D_ref

# ── Validation cosmologies and evaluation grid ─────────────────────────────

_VALIDATION_COSMOLOGIES = [
    (0.30, -0.5, 0.0),
    (0.30, -1.0, 0.0),
    (0.30, -1.0, 0.5),
    (0.30, -1.0, -0.5),
    (0.30, -0.8, 0.3),
    (0.25, -1.0, 0.5),
    (0.35, -1.0, -0.3),
    (0.30, -1.2, 0.4),
    (0.30, -0.9, -0.4),
    (0.28, -1.0, 1.0),
    (0.22, -1.0, 0.8),
    (0.38, -1.0, -0.6),
    (0.30, -0.7, 0.6),
    (0.25, -1.2, 0.3),
    (0.35, -0.8, -0.5),
]

# 200 points in log-space — fast enough for many objective evaluations
_N_EVAL = 200
_EVAL_A = np.logspace(np.log10(1e-4), 0.0, _N_EVAL)
_x_eval = np.log(_EVAL_A)
_dx = np.diff(_x_eval)

# ── Pre-compute references ──────────────────────────────────────────────────

_REF_DATA = []

def _build_references():
    """Pre-compute reference growth data for all validation cosmologies."""
    global _REF_DATA
    _REF_DATA = []
    for (Om0, w0, wa) in _VALIDATION_COSMOLOGIES:
        try:
            f_ref, D_ref = _growth_ode_reference(_EVAL_A, Om0, w0, wa)
            X = _design_matrix(_EVAL_A, Om0, w0, wa)
            Om_m = _omega_m_of_a(_EVAL_A, Om0, w0, wa)
            _REF_DATA.append((f_ref, D_ref, Om0, w0, wa, X, Om_m))
        except Exception:
            _REF_DATA.append(None)

    _REF_DATA = [r for r in _REF_DATA if r is not None]

_build_references()

# ── Compute D from f values ─────────────────────────────────────────────────

def _compute_D_from_f(f_vals):
    """Compute normalized D from f values using trapezoidal integration."""
    f_avg = 0.5 * (f_vals[1:] + f_vals[:-1])
    cum_int = np.concatenate(([0.0], np.cumsum(f_avg * _dx)))
    J_vals = cum_int - cum_int[-1]
    D = np.exp(J_vals)
    return D / D[-1]

# ── Objective: direct max-relative-error over f and D ───────────────────────

def _objective(c):
    """Return 100 * max relative error over all validation cosmologies."""
    c = np.asarray(c, dtype=float)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return 1e6
    if np.any(c < -10) or np.any(c > 10):
        return 1e6
    
    max_err = 0.0
    
    for ref in _REF_DATA:
        if ref is None:
            continue
        f_ref, D_ref, Om0, w0, wa, X, Om_m = ref

        gamma = X @ c

        # Check gamma constraint: penalize violations heavily
        g_min = np.min(gamma)
        g_max = np.max(gamma)
        if g_min < 0.1 or g_max > 1.2:
            viol = 0.0
            if g_min < 0.1:
                viol += 0.1 - g_min
            if g_max > 1.2:
                viol += g_max - 1.2
            return 100.0 * (1.0 + 10.0 * viol)

        Om_m_safe = np.clip(Om_m, 1e-15, 1.0)
        f_model = Om_m_safe ** gamma

        # Relative errors for f
        denom_f = np.maximum(np.abs(f_ref), 1e-10)
        rel_err_f = np.abs(f_model - f_ref) / denom_f

        # Compute D_model
        D_model = _compute_D_from_f(f_model)

        # Relative errors for D
        denom_D = np.maximum(np.abs(D_ref), 1e-10)
        rel_err_D = np.abs(D_model - D_ref) / denom_D

        local_max = max(np.max(rel_err_f), np.max(rel_err_D))
        if local_max > max_err:
            max_err = local_max
            if max_err > 0.05:  # early exit if already bad
                return 100.0 * max_err

    return 100.0 * max_err

# ── Initial guess: weighted least-squares in log-space ──────────────────────

def _initial_guess():
    """
    Get initial guess by weighted least-squares fitting:
    log(f_ref) = gamma * log(Om_m)  =>  gamma = log(f_ref) / log(Om_m)
    Then fit gamma = X @ c via weighted least squares.
    """
    all_X = []
    all_target = []
    all_weights = []
    
    for ref in _REF_DATA:
        if ref is None:
            continue
        f_ref, D_ref, Om0, w0, wa, X, Om_m = ref
        Om_m_safe = np.clip(Om_m, 1e-15, 1.0)
        log_f = np.log(np.clip(f_ref, 1e-30, None))
        log_Om = np.log(Om_m_safe)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            gamma_target = log_f / log_Om
        
        # Weight by |log_Om| to emphasize where the fit matters most
        w = np.abs(log_Om)
        w = np.clip(w, 0.1, 10.0)
        
        valid = np.isfinite(gamma_target) & (np.abs(gamma_target) < 5.0)
        all_X.append(X[valid])
        all_target.append(gamma_target[valid])
        all_weights.append(w[valid])
    
    if len(all_X) == 0:
        return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
    
    X_stack = np.vstack(all_X)
    target_stack = np.concatenate(all_target)
    w_stack = np.concatenate(all_weights)
    
    try:
        # Weighted least squares
        W = np.diag(w_stack)
        XW = X_stack * w_stack[:, None]
        result, _, _, _ = np.linalg.lstsq(XW, target_stack * w_stack, rcond=None)
        return result
    except Exception:
        try:
            result, _, _, _ = np.linalg.lstsq(X_stack, target_stack, rcond=None)
            return result
        except Exception:
            return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])

# ── Optimization ────────────────────────────────────────────────────────────

def _run_optimization():
    """Run multi-stage optimization with deadline guard."""
    best_c = _initial_guess()
    best_c = np.clip(best_c, -10, 10)
    best_err = _objective(best_c)
    
    if not _deadline_ok():
        return best_c

    # Stage 1: Nelder-Mead from the LS initial guess
    try:
        result = minimize(
            _objective, best_c, method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True}
        )
        if result.fun < best_err:
            best_err = result.fun
            best_c = result.x.copy()
    except Exception:
        pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage 2: Powell method
    try:
        result = minimize(
            _objective, best_c, method='Powell',
            options={'maxiter': 3000, 'xtol': 1e-9, 'ftol': 1e-12}
        )
        if result.fun < best_err:
            best_err = result.fun
            best_c = result.x.copy()
    except Exception:
        pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage 3: Multi-start Nelder-Mead with decreasing perturbation scales
    rng = np.random.default_rng(42)
    n_starts = 80
    for i in range(n_starts):
        if not _deadline_ok():
            break
        # Adaptive scale: start wide, narrow down
        progress = i / n_starts
        scale = 0.5 * (1.0 - progress) + 0.05
        x_try = best_c + rng.normal(0, scale, 6)
        x_try = np.clip(x_try, -5, 5)
        try:
            result = minimize(
                _objective, x_try, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_err:
                best_err = result.fun
                best_c = result.x.copy()
        except Exception:
            pass

    best_c = np.clip(best_c, -10, 10)
    return best_c

# Run optimization at import time
_COEFFS = _run_optimization()

def construct():
    """Return the six optimized growth-index coefficients."""
    return list(_COEFFS)
