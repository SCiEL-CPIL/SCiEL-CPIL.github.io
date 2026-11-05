"""Compact dark-energy growth-index fit: smooth log-sum-exp surrogate + scipy L-BFGS-B multi-start."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize

# Fixed internal deadline with a safety margin (evaluator kills at 90 s).
_DEADLINE = 75.0

# Cosmology grid: a in (0, 1] mapped to x = ln a in [-3, 0].
_A_GRID = np.exp(np.linspace(-3.0, 0.0, 240))
_X_GRID = np.log(_A_GRID)
_DX = np.diff(_X_GRID)


def _omega_m(a, om0, w0, wa):
    a2 = a * a
    a3 = a2 * a
    return om0 / (om0 + (1.0 - om0) * a3 * np.exp(3.0 * (w0 + 1.0 + wa) * (1.0 - a) / (1.0 + wa)))


def _gamma_of(c, a, om0, w0, wa):
    om = _omega_m(a, om0, w0, wa)
    q = 1.0 - om
    u = 1.0 + w0 + wa * (1.0 - a)
    return (c[0] + c[1] * q + c[2] * q * q + c[3] * u + c[4] * wa * a + c[5] * u * q)


def _f_of(c, a, om0, w0, wa):
    om = _omega_m(a, om0, w0, wa)
    g = _gamma_of(c, a, om0, w0, wa)
    return np.power(np.maximum(om, 1e-15), g)


def _norm_d_of(c, om0, w0, wa):
    f = _f_of(c, _A_GRID, om0, w0, wa)
    integ = np.zeros_like(_X_GRID)
    integ[1:] = np.cumsum(0.5 * (f[1:] + f[:-1]) * _DX)
    total = integ[-1]
    val = total - integ
    return np.exp(-val)


def _build_reference(om0, w0, wa):
    """Solve the GR growth Riccati for f(a) and integrate for D(a)."""
    N = 240
    x = np.linspace(-3.0, 0.0, N)
    a = np.exp(x)
    om = _omega_m(a, om0, w0, wa)
    da = np.diff(a)
    om_prime = np.zeros_like(a)
    om_prime[1:] = (om[1:] - om[:-1]) / da
    om_prime[0] = om_prime[1]
    rhs = 1.5 * a * (om + om_prime)
    Acoef = (a - 1.0)
    disc = 1.0 + 4.0 * Acoef * rhs
    disc = np.maximum(disc, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = (-1.0 + np.sqrt(disc)) / (2.0 * Acoef)
    g[0] = 0.5 * om[0]
    lnD = np.zeros_like(x)
    lnD[1:] = np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(x))
    D = np.exp(lnD)
    D = D / D[-1]
    return a, g, D


def _compute_all_errors(c, models, ref_cache):
    """Return concatenated relative error vector for f and D across all models."""
    errs = []
    for m in models:
        om0, w0, wa = m
        a, ref_f, ref_D = ref_cache[m]
        g = _gamma_of(c, _A_GRID, om0, w0, wa)
        f = _f_of(c, _A_GRID, om0, w0, wa)
        D = _norm_d_of(c, om0, w0, wa)
        er_f = np.abs((f - ref_f) / np.maximum(ref_f, 1e-12))
        er_D = np.abs((D - ref_D) / np.maximum(np.abs(ref_D), 1e-12))
        errs.append(er_f)
        errs.append(er_D)
    return np.concatenate(errs)


def _smooth_objective(c, models, ref_cache, beta=50.0):
    """Log-sum-exp surrogate for max relative error, plus gamma feasibility penalty."""
    # Check gamma feasibility
    for m in models:
        om0, w0, wa = m
        g = _gamma_of(c, _A_GRID, om0, w0, wa)
        viol_lo = np.maximum(0.1 - g, 0.0)
        viol_hi = np.maximum(g - 1.2, 0.0)
        if np.any(viol_lo > 0) or np.any(viol_hi > 0):
            penalty = float(np.max(viol_lo) + np.max(viol_hi))
            return 1.0 + penalty * 10.0

    errs = _compute_all_errors(c, models, ref_cache)
    if len(errs) == 0:
        return 1.0
    # log-sum-exp for smooth max
    m = np.max(errs)
    if m < 1e-15:
        return 0.0
    lse = m + np.log(np.sum(np.exp((errs - m) * beta)) / len(errs)) / beta
    return float(lse)


def construct():
    t0 = time.time()
    models = [
        (0.30, -0.9, 0.0),
        (0.28, -1.0, 0.0),
        (0.31, -1.1, 0.3),
        (0.27, -0.8, -0.3),
        (0.30, -1.0, 0.5),
        (0.33, -0.95, 0.2),
    ]

    # Pre-compute references
    ref_cache = {}
    for m in models:
        ref_cache[m] = _build_reference(*m)

    def true_obj(c):
        """Exact max relative error (for final evaluation)."""
        worst = 0.0
        for m in models:
            om0, w0, wa = m
            a, ref_f, ref_D = ref_cache[m]
            g = _gamma_of(c, _A_GRID, om0, w0, wa)
            if not (np.all(g >= 0.1) and np.all(g <= 1.2)):
                viol = np.maximum(0.1 - g, 0.0) + np.maximum(g - 1.2, 0.0)
                return 1.0 + float(np.max(viol))
            f = _f_of(c, _A_GRID, om0, w0, wa)
            D = _norm_d_of(c, om0, w0, wa)
            er_f = np.max(np.abs((f - ref_f) / np.maximum(ref_f, 1e-12)))
            er_D = np.max(np.abs((D - ref_D) / np.maximum(np.abs(ref_D), 1e-12)))
            worst = max(worst, er_f, er_D)
        return float(worst)

    bounds = [(-10.0, 10.0)] * 6

    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_val = true_obj(best_c)

    rng = np.random.default_rng(12345)
    max_starts = 40
    start_idx = 0

    while time.time() < _DEADLINE - 10.0 and start_idx < max_starts:
        start_idx += 1
        c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
        c0 += rng.normal(0.0, 0.2, size=6)
        c0 = np.clip(c0, -10.0, 10.0)

        # Use smooth surrogate for optimization
        smooth_obj = lambda x: _smooth_objective(x, models, ref_cache, beta=50.0)
        try:
            res = minimize(smooth_obj, c0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            c_candidate = np.clip(res.x, -10.0, 10.0)
        except Exception:
            c_candidate = c0

        # Evaluate with true objective
        val = true_obj(c_candidate)
        if val < best_val:
            best_val = val
            best_c = c_candidate

        # Also try a few local refinements around the best point found so far
        if start_idx % 5 == 0 and time.time() < _DEADLINE - 15.0:
            for _ in range(3):
                c_try = best_c + rng.normal(0.0, 0.05, size=6)
                c_try = np.clip(c_try, -10.0, 10.0)
                try:
                    res2 = minimize(smooth_obj, c_try, method='L-BFGS-B', bounds=bounds,
                                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                    c_try = np.clip(res2.x, -10.0, 10.0)
                except Exception:
                    pass
                val2 = true_obj(c_try)
                if val2 < best_val:
                    best_val = val2
                    best_c = c_try

    # Final safety
    if not np.all(np.isfinite(best_c)):
        best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_c = np.clip(best_c, -10.0, 10.0)
    return [float(x) for x in best_c]
# EVOLVE-BLOCK-END
