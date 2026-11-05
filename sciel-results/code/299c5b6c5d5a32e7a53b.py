"""Growth-index approximation for CPL dark-energy cosmologies.

All heavy computation (reference ODE solving + coefficient optimization) is
performed at module import time.  construct() returns pre-computed coefficients.

Public task contract:
    f = Omega_m(a) ** gamma
    D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)

    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    q = 1 - Omega_m(a)
    u = 1 + w0 + wa*(1 - a)

    gamma must lie in [0.1, 1.2] at all checked locations.
    Coefficients must be in [-10, 10].
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

# ── Deadline guard ──────────────────────────────────────────────────────────
_START_TIME = time.time()
_TIME_LIMIT = 85.0  # 5 s safety margin on a 90 s wall clock

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
    Om_m = Om0 * a_inv3 / (Om0 * a_inv3 + Om_DE)
    return Om_m

def _gamma_from_coeffs(c, a, Om0, w0, wa):
    """Compute gamma from the six-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = c
    Om_m = _omega_m_of_a(a, Om0, w0, wa)
    q = 1.0 - Om_m
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    return gamma

def _f_from_gamma(gamma, Om_m):
    """f = Om_m^gamma."""
    Om_m_safe = np.clip(Om_m, 1e-15, 1.0)
    return Om_m_safe ** gamma

# ── Reference: solve the growth ODE numerically in x = ln(a) space ─────────

def _growth_ode_reference(a_grid, Om0, w0, wa):
    """Compute f(a) and D(a)/D(1) by solving the linear growth ODE in x-space."""
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
        D_prime = F
        D_double = -(1.5 + (1.0 + w) - 1.5 * dOm_m_dx) * F + 1.5 * Om_m * D
        return [D_prime, D_double]

    D0 = a_min ** (5.0 / 6.0)
    F0 = (5.0 / 6.0) * a_min ** (-1.0 / 6.0)

    try:
        sol = solve_ivp(ode, [x_min, x_max], [D0, F0],
                        t_eval=a_grid,
                        max_step=0.005, rtol=1e-10, atol=1e-12)
        if not sol.success:
            raise RuntimeError("ODE integration failed")

        D_ref = sol.y[0]
        F_ref = sol.y[1]
        D_ref = D_ref / D_ref[-1]
        mask = np.isfinite(F_ref) & np.isfinite(D_ref) & (D_ref > 0)
        if np.sum(mask) < 10:
            raise RuntimeError("Too few valid points")
        return F_ref[mask], D_ref[mask]
    except Exception:
        # Fallback: constant gamma reference
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

# ── Validation cosmologies and evaluation grid ──────────────────────────────

_VALIDATION_COSMOLOGIES = [
    (0.30, -1.0, 0.0),
    (0.30, -1.0, 0.3),
    (0.30, -1.0, -0.3),
    (0.30, -0.5, 0.0),
    (0.30, -1.5, 0.0),
    (0.25, -1.0, 0.0),
    (0.35, -1.0, 0.0),
    (0.30, -0.5, 0.5),
    (0.30, -1.0, 0.5),
    (0.30, -1.0, -0.5),
    (0.30, -0.8, 0.3),
    (0.25, -1.0, 0.5),
    (0.35, -1.0, -0.3),
    (0.30, -1.2, 0.4),
    (0.30, -0.9, -0.4),
    (0.28, -1.0, 1.0),
    (0.32, -0.7, -0.6),
    (0.27, -1.3, 0.8),
]

_EVAL_A = np.logspace(np.log10(0.02), 0.0, 200)
_EVAL_X = np.log(_EVAL_A)
_DX = np.diff(_EVAL_X)

# ── Pre-compute reference data at import time ───────────────────────────────

_REF_DATA = []
for _Om0, _w0, _wa in _VALIDATION_COSMOLOGIES:
    if not _deadline_ok():
        break
    try:
        _f_ref, _D_ref = _growth_ode_reference(_EVAL_A, _Om0, _w0, _wa)
        _REF_DATA.append((_f_ref, _D_ref, _Om0, _w0, _wa))
    except Exception:
        pass

# ── Objective function ──────────────────────────────────────────────────────

def _objective(c):
    """Compute max relative error over all validation cosmologies."""
    c = np.asarray(c, dtype=float)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return 1e6
    if np.any(c < -10) or np.any(c > 10):
        return 1e6

    max_err = 0.0
    a_eval = _EVAL_A
    x_eval = _EVAL_X

    for idx, ref in enumerate(_REF_DATA):
        f_ref, D_ref, Om0, w0, wa = ref

        Om_m_eval = _omega_m_of_a(a_eval, Om0, w0, wa)
        gamma_eval = _gamma_from_coeffs(c, a_eval, Om0, w0, wa)

        if np.any(gamma_eval < 0.1) or np.any(gamma_eval > 1.2):
            gamma_violation = np.maximum(0.0, 0.1 - gamma_eval) + np.maximum(0.0, gamma_eval - 1.2)
            max_err = max(max_err, 10.0 * np.max(gamma_violation))
            continue

        f_model = _f_from_gamma(gamma_eval, Om_m_eval)

        # Relative error in f
        rel_f = np.abs(f_model - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        err_f = np.max(rel_f)
        max_err = max(max_err, err_f)

        # Compute D_model from f_model using cumulative integral in x-space
        # D(a)/D(1) = exp( integral from 0 to ln(a) of f(exp(x)) dx )
        # Since x is increasing from x_min to 0:
        # C(x_i) = integral from x_min to x_i of f dx
        # integral from 0 to x_i = C(x_i) - C(0) = C(x_i) - C(x_max)
        f_avg = 0.5 * (f_model[1:] + f_model[:-1])
        cum_int = np.concatenate(([0.0], np.cumsum(f_avg * _DX)))
        integral_vals = cum_int - cum_int[-1]
        D_model = np.exp(integral_vals)
        D_model = D_model / D_model[-1]

        rel_D = np.abs(D_model - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
        err_D = np.max(rel_D)
        max_err = max(max_err, err_D)

    return max_err

# ── Optimization at import time ─────────────────────────────────────────────

def _run_optimization():
    """Multi-start Nelder-Mead with structured initialization and perturbation refinement."""
    best_coeff = np.array([0.55, 0.0, 0.0, -0.1, 0.0, 0.0])
    best_score = np.inf

    starts = [
        [0.55, 0.0, 0.0, -0.1, 0.0, 0.0],
        [0.55, 0.1, 0.0, -0.2, 0.0, 0.0],
        [0.55, -0.1, 0.0, 0.1, 0.0, 0.0],
        [0.55, 0.0, 0.1, -0.1, 0.0, 0.0],
        [0.55, 0.0, 0.0, -0.1, 0.1, 0.0],
        [0.55, 0.0, 0.0, -0.1, 0.0, 0.1],
        [0.55, 0.05, 0.05, -0.15, 0.05, 0.05],
        [0.55, -0.05, 0.05, -0.05, -0.05, 0.05],
        [0.52, 0.15, -0.05, -0.1, 0.05, -0.05],
        [0.58, -0.1, 0.1, -0.05, -0.05, 0.1],
        [0.50, 0.2, 0.0, -0.15, 0.1, 0.05],
        [0.60, -0.15, 0.05, -0.08, -0.05, -0.08],
    ]

    for start in starts:
        if not _deadline_ok():
            break

        def obj(coeffs):
            if not _deadline_ok():
                return 1e6
            return _objective(coeffs)

        try:
            res = minimize(obj, start, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_coeff = res.x.copy()
        except Exception:
            continue

    # Perturbation-based local refinement
    if _deadline_ok():
        rng = np.random.default_rng(42)
        for perturb_scale in [0.02, 0.05, 0.1, 0.2]:
            if not _deadline_ok():
                break
            for _ in range(4):
                if not _deadline_ok():
                    break
                trial = best_coeff + perturb_scale * rng.standard_normal(6)
                trial = np.clip(trial, -10, 10)

                def obj(coeffs):
                    if not _deadline_ok():
                        return 1e6
                    return _objective(coeffs)

                try:
                    res = minimize(obj, trial, method='Nelder-Mead',
                                  options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
                    if res.fun < best_score:
                        best_score = res.fun
                        best_coeff = res.x.copy()
                except Exception:
                    continue

    return np.clip(best_coeff, -10.0, 10.0)

# Run optimization at import time
_PRECOMPUTED_COEFFS = _run_optimization()

def construct():
    """Return the pre-computed six coefficients."""
    return _PRECOMPUTED_COEFFS.tolist()
