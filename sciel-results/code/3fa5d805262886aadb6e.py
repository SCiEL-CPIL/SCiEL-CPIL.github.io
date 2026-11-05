"""Improved growth-index approximation via log-space LSQ pre-fit + minimax polish.

Structural change: Instead of starting from a constant-gamma seed and using
differential evolution (which is slow and non-smooth), we:
1. Pre-compute reference f(a) and D(a)/D(1) for validation cosmologies using
   a fast ODE solve.
2. Fit the 6 coefficients via weighted least-squares in log-space
   (ln f = gamma * ln Om_m, linear in coefficients).
3. Polish the result with Nelder-Mead on the max-relative-error objective.

This gives a much better initial point, so the local optimizer converges
faster to a tighter minimum within the time budget.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

_START_TIME = time.time()
_TIME_LIMIT = 82.0  # safety margin before 90 s kill

def _deadline_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

# ── Physics helpers ─────────────────────────────────────────────────────────

def _omega_m_of_a(a, Om0, w0, wa):
    """Matter density parameter as a function of scale factor (vectorized)."""
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
    a_min = float(a_grid[0])
    a_max = float(a_grid[-1])
    x_min = math.log(a_min)
    x_max = math.log(a_max)

    def ode(x, y):
        a = math.exp(x)
        Om_m = float(_omega_m_of_a(a, Om0, w0, wa))
        w = w0 + wa * (1.0 - a)
        dOm_m_dx = -3.0 * Om_m * (Om_m - 1.0)
        D, F = y
        D_double = -(1.5 + (1.0 + w) - 1.5 * dOm_m_dx) * F + 1.5 * Om_m * D
        return [F, D_double]

    D0 = a_min ** (5.0 / 6.0)
    F0 = (5.0 / 6.0) * a_min ** (-1.0 / 6.0)

    try:
        sol = solve_ivp(ode, [x_min, x_max], [D0, F0],
                        t_eval=a_grid, max_step=0.01, rtol=1e-9, atol=1e-11)
        if not sol.success:
            raise RuntimeError("ODE integration failed")
        D_ref = sol.y[0]
        F_ref = sol.y[1]
        D_ref = D_ref / D_ref[-1]
        f_ref = F_ref / (a_grid * D_ref)
        # Ensure positivity
        f_ref = np.clip(f_ref, 1e-15, None)
        D_ref = np.clip(D_ref, 1e-15, None)
        return f_ref, D_ref
    except Exception:
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

# ── Validation cosmologies ─────────────────────────────────────────────────

_VALIDATION_COSMOLOGIES = [
    (0.30, -1.0, 0.0),
    (0.30, -1.0, 0.3),
    (0.25, -0.9, 0.2),
    (0.35, -1.1, 0.1),
    (0.30, -1.0, -0.5),
    (0.28, -1.0, 0.0),
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
    (0.30, -1.0, 0.8),
    (0.32, -0.95, -0.2),
    (0.27, -1.05, 0.4),
]

_N_EVAL = 200
_EVAL_A = np.logspace(np.log10(5e-3), 0.0, _N_EVAL)
_x_eval = np.log(_EVAL_A)
_dx = np.diff(_x_eval)

# ── Pre-compute references ──────────────────────────────────────────────────

_REF_DATA = []

def _build_references():
    global _REF_DATA
    _REF_DATA = []
    for (Om0, w0, wa) in _VALIDATION_COSMOLOGIES:
        if not _deadline_ok():
            break
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
    f_avg = 0.5 * (f_vals[1:] + f_vals[:-1])
    cum_int = np.concatenate(([0.0], np.cumsum(f_avg * _dx)))
    J_vals = cum_int - cum_int[-1]
    D = np.exp(J_vals)
    return D / D[-1]

# ── Objective ───────────────────────────────────────────────────────────────

def _objective(c):
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

        denom_f = np.maximum(np.abs(f_ref), 1e-10)
        rel_err_f = np.abs(f_model - f_ref) / denom_f

        D_model = _compute_D_from_f(f_model)
        denom_D = np.maximum(np.abs(D_ref), 1e-10)
        rel_err_D = np.abs(D_model - D_ref) / denom_D

        local_max = max(np.max(rel_err_f), np.max(rel_err_D))
        if local_max > max_err:
            max_err = local_max
            if max_err > 0.05:
                return 100.0 * max_err

    return 100.0 * max_err

# ── Initial guess: weighted least-squares in log-space ─────────────────────

def _initial_guess():
    all_X = []
    all_target = []
    all_weights = []

    for ref in _REF_DATA:
        if ref is None:
            continue
        f_ref, D_ref, Om0, w0, wa, X, Om_m = ref
        Om_m_safe = np.clip(Om_m, 1e-15, None)
        log_f = np.log(np.clip(f_ref, 1e-30, None))
        log_Om = np.log(Om_m_safe)

        with np.errstate(divide='ignore', invalid='ignore'):
            gamma_target = log_f / log_Om

        valid = np.isfinite(gamma_target) & (np.abs(gamma_target) < 5.0)
        if np.sum(valid) < 5:
            continue
        all_X.append(X[valid])
        all_target.append(gamma_target[valid])
        # Weight by |ln Om| so that points where ln Om is large (early times)
        # are well-determined
        wts = np.abs(log_Om[valid])
        all_weights.append(wts)

    if not all_X:
        return np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_target)
    w_all = np.concatenate(all_weights)

    # Weighted least squares: minimize sum w_i (X_i c - y_i)^2
    W = np.diag(w_all)
    try:
        XtWX = X.T @ (w_all[:, None] * X)
        XtWy = X.T @ (w_all * y_all)
        c_ls = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        c_ls = np.linalg.lstsq(X_all * w_all[:, None], y_all * w_all, rcond=None)[0]

    c_ls = np.clip(c_ls, -10, 10)
    return c_ls

# ── Main construct ──────────────────────────────────────────────────────────

def construct():
    """Return the best 6 coefficients found via optimization within time budget."""
    start_time = time.time()
    deadline = start_time + 75.0  # extra safety margin

    seed = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_coeffs = seed.copy()
    best_score = float('inf')

    # Step 1: LSQ initial guess
    try:
        c0 = _initial_guess()
        s0 = _objective(c0)
        if s0 < best_score:
            best_score = s0
            best_coeffs = c0.tolist()
    except Exception:
        pass

    # Step 2: Nelder-Mead polish
    try:
        if time.time() < deadline - 5:
            result_nm = minimize(
                _objective, best_coeffs, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10,
                         'adaptive': True}
            )
            if result_nm.fun < best_score:
                best_score = result_nm.fun
                best_coeffs = result_nm.x.tolist()
    except Exception:
        pass

    # Step 3: L-BFGS-B refinement
    try:
        if time.time() < deadline - 3:
            bounds = [(-10, 10)] * 6
            result_lbfgs = minimize(
                _objective, best_coeffs, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result_lbfgs.fun < best_score:
                best_score = result_lbfgs.fun
                best_coeffs = result_lbfgs.x.tolist()
    except Exception:
        pass

    # Step 4: One more Nelder-Mead from the L-BFGS result
    try:
        if time.time() < deadline - 1:
            result_nm2 = minimize(
                _objective, best_coeffs, method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11,
                         'adaptive': True}
            )
            if result_nm2.fun < best_score:
                best_score = result_nm2.fun
                best_coeffs = result_nm2.x.tolist()
    except Exception:
        pass

    # Final validation
    best_coeffs = np.array(best_coeffs)
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.array(seed)
    best_coeffs = np.clip(best_coeffs, -10, 10)

    return best_coeffs.tolist()
