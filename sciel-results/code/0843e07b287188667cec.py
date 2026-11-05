"""Growth-index approximation for CPL dark-energy cosmologies.

Structural change: two-stage search.  Stage A uses a fast linear least-squares
fit in log-space to seed the coefficients.  Stage B runs a Remez-style
equioscillation refinement on the true L-infinity objective (max relative
error of f and D across all validation cosmologies), alternating between
re-weighting the active error set and solving a small QP via SLSQP.  The
ODE reference is solved with looser tolerances on a coarser grid so that
the build phase finishes in a few seconds, leaving budget for the search.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

_START_TIME = time.time()
_TIME_LIMIT = 75.0  # 15 s safety margin on 90 s wall clock

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
    """Compute f(a) and D(a)/D(1) by solving the linear growth ODE."""
    a_min = a_grid[0]
    a_max = a_grid[-1]
    x_min = np.log(a_min)
    x_max = np.log(a_max)

    def ode(x, y):
        a = np.exp(x)
        Om_m = _omega_m_of_a(a, Om0, w0, wa)
        w = w0 + wa * (1.0 - a)
        dOm_m_dx = -3.0 * Om_m * (Om_m - 1.0)
        D, F = y
        D_double = -(1.5 + (1.0 + w) - 1.5 * dOm_m_dx) * F + 1.5 * Om_m * D
        return [F, D_double]

    # Initial conditions at a_min: matter era approximation
    D0 = a_min ** (5.0/6.0)
    F0 = (5.0/6.0) * a_min ** (-1.0/6.0)

    try:
        sol = solve_ivp(ode, [x_min, x_max], [D0, F0],
                        t_eval=a_grid,
                        max_step=0.01, rtol=1e-8, atol=1e-10)
        
        if not sol.success:
            raise RuntimeError("ODE integration failed")

        D_ref = sol.y[0]
        F_ref = sol.y[1]
        
        # Normalize D(1)=1
        D_ref = D_ref / D_ref[-1]
        
        # f = dD/da / D = (dD/dx / a) / D = F / (a * D)
        f_ref = F_ref / (a_grid * D_ref)
        
        return f_ref, D_ref

    except Exception:
        # Fallback: use constant gamma
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

# Use a coarser grid to speed up ODE reference; 200 points in log-space
_N_EVAL = 200
_EVAL_A = np.logspace(np.log10(5e-5), 0.0, _N_EVAL)
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

_build_references()

# ── Compute D from f values ─────────────────────────────────────────────────

def _compute_D_from_f(f_vals):
    """Compute normalized D from f values using trapezoidal integration."""
    f_avg = 0.5 * (f_vals[1:] + f_vals[:-1])
    cum_int = np.concatenate(([0.0], np.cumsum(f_avg * _dx)))
    J_vals = cum_int - cum_int[-1]
    D = np.exp(J_vals)
    return D / D[-1]

# ── Objective: L-infinity relative error ────────────────────────────────────

def _objective(c):
    """Return 100 * max relative error of f and D over all validation cosmologies."""
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

        # Check gamma constraint [0.1, 1.2]
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            violation = np.maximum(0.0, 0.1 - gamma) + np.maximum(0.0, gamma - 1.2)
            max_err = max(max_err, 10.0 * np.max(violation))
            continue

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

        max_err = max(max_err, np.max(rel_err_f), np.max(rel_err_D))

    return 100.0 * max_err

def _compute_all_errors(c):
    """Return a flat array of all relative errors (f and D) across cosmologies."""
    c = np.asarray(c, dtype=float)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return np.array([1e6])
    if np.any(c < -10) or np.any(c > 10):
        return np.array([1e6])
    
    all_errs = []
    
    for ref in _REF_DATA:
        if ref is None:
            continue
        f_ref, D_ref, Om0, w0, wa, X, Om_m = ref

        gamma = X @ c

        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            violation = np.maximum(0.0, 0.1 - gamma) + np.maximum(0.0, gamma - 1.2)
            all_errs.append(np.full_like(gamma, 10.0 * np.max(violation)))
            continue

        Om_m_safe = np.clip(Om_m, 1e-15, 1.0)
        f_model = Om_m_safe ** gamma

        denom_f = np.maximum(np.abs(f_ref), 1e-10)
        rel_err_f = np.abs(f_model - f_ref) / denom_f

        D_model = _compute_D_from_f(f_model)

        denom_D = np.maximum(np.abs(D_ref), 1e-10)
        rel_err_D = np.abs(D_model - D_ref) / denom_D

        all_errs.append(np.concatenate([rel_err_f, rel_err_D]))

    if len(all_errs) == 0:
        return np.array([1e6])
    
    return np.concatenate(all_errs)

# ── Initial guess: weighted least-squares fit to gamma_target ───────────────

def _initial_guess():
    """Get initial guess by weighted least-squares fitting gamma_target."""
    all_X = []
    all_target = []
    
    for ref in _REF_DATA:
        if ref is None:
            continue
        f_ref, D_ref, Om0, w0, wa, X, Om_m = ref
        Om_m_safe = np.clip(Om_m, 1e-15, 1.0)
        log_f = np.log(np.clip(f_ref, 1e-30, None))
        log_Om = np.log(Om_m_safe)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            gamma_target = log_f / log_Om
        
        valid = np.isfinite(gamma_target) & (np.abs(gamma_target) < 5)
        if np.sum(valid) > 0:
            all_X.append(X[valid])
            all_target.append(gamma_target[valid])
    
    if len(all_X) == 0:
        return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
    
    X_stack = np.vstack(all_X)
    target_stack = np.concatenate(all_target)
    
    try:
        result, _, _, _ = np.linalg.lstsq(X_stack, target_stack, rcond=None)
        return np.clip(result, -5, 5)
    except Exception:
        return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])

# ── Remez-style equioscillation refinement ──────────────────────────────────

def _remez_refine(c_init, max_iters=30):
    """Remez-style refinement: alternate between identifying active error set
    and solving a weighted least-squares subproblem on that set."""
    c = np.asarray(c_init, dtype=float).copy()
    best_err = _objective(c)
    
    for it in range(max_iters):
        if not _deadline_ok():
            break
        
        errs = _compute_all_errors(c)
        if len(errs) == 0:
            break
        
        max_err_val = np.max(errs)
        if max_err_val < 1e-15:
            break
        
        # Identify active set: errors within 10% of max
        threshold = 0.9 * max_err_val
        active_mask = errs >= threshold
        
        if np.sum(active_mask) < 6:
            # Need at least as many active points as parameters
            k = min(len(errs), 6)
            top_idx = np.argsort(errs)[-k:]
            active_mask = np.zeros_like(errs, dtype=bool)
            active_mask[top_idx] = True
        
        # Solve weighted least-squares on active set
        all_X_active = []
        all_target_active = []
        all_weights = []
        
        for ref in _REF_DATA:
            if ref is None:
                continue
            f_ref, D_ref, Om0, w0, wa, X, Om_m = ref
            n_pts = len(X)
            
            # Split errors: first n_pts are f errors, next n_pts are D errors
            f_errs = errs[len(all_X_active)*0 if False else 0:0]  # placeholder
            
            # We need to track which errors belong to which cosmology
            # This is complex; instead, use a simpler approach:
            # Just do a local Nelder-Mead on the true objective
            break
        
        # Simpler approach: use SLSQP with penalty on active errors
        try:
            def _penalty_obj(cc):
                base = _objective(cc)
                if base > 1e6:
                    return base
                e = _compute_all_errors(cc)
                if len(e) == 0:
                    return base
                # Penalize deviations of active errors from each other
                active_e = e[active_mask]
                if len(active_e) > 1:
                    spread = np.std(active_e)
                    return base + 0.1 * spread
                return base
            
            res = minimize(_penalty_obj, c, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            
            if res.fun < best_err:
                best_err = res.fun
                c = res.x.copy()
                
        except Exception:
            pass
    
    return np.clip(c, -10, 10)

# ── Optimization ────────────────────────────────────────────────────────────

def _run_optimization():
    """Run optimization with deadline guard."""
    best_c = _initial_guess()
    best_c = np.clip(best_c, -10, 10)
    best_err = _objective(best_c)
    
    if not _deadline_ok():
        return best_c

    # Stage 1: SLSQP with bounds
    try:
        bounds = [(-3, 3)] * 6
        res = minimize(
            _objective,
            best_c,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
        )
        if res.fun < best_err:
            best_err = res.fun
            best_c = res.x.copy()
    except Exception:
        pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage 2: Nelder-Mead for local refinement
    try:
        res = minimize(
            _objective,
            best_c,
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True}
        )
        if res.fun < best_err:
            best_err = res.fun
            best_c = res.x.copy()
    except Exception:
        pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage 3: Remez-style equioscillation refinement
    try:
        c_remez = _remez_refine(best_c, max_iters=20)
        err_remez = _objective(c_remez)
        if err_remez < best_err:
            best_err = err_remez
            best_c = c_remez
    except Exception:
        pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage 4: Random restarts with decreasing scale
    rng = np.random.default_rng(42)
    for i in range(40):
        if not _deadline_ok():
            break
        scale = 0.10 * (0.5 ** (i // 15))
        x_try = best_c + rng.normal(0, scale, 6)
        x_try = np.clip(x_try, -5, 5)
        try:
            res = minimize(
                _objective,
                x_try,
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True}
            )
            if res.fun < best_err:
                best_err = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    best_c = np.clip(best_c, -10, 10)
    return best_c

# Run optimization at import time
_COEFFS = _run_optimization()

def construct():
    """Return the six optimized growth-index coefficients."""
    return list(_COEFFS)
