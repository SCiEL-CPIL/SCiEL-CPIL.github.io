"""Cosmology: compact dark-energy growth approximation.

Fits the six-coefficient growth-index ansatz to a tightly integrated
general-relativistic second-order ODE reference for CPL dark-energy
cosmologies.  The score is 100 * max relative error of f and the
normalized growth factor D over disjoint validation cosmologies.
"""
# EVOLVE-BLOCK-START
import math
import time
import random
import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Reference cosmology machinery (CPL dark energy)
# ---------------------------------------------------------------------------

def _H2(a, Om0, w0, wa):
    """H^2(a) in units of H0^2 for CPL dark energy."""
    de = Om0 * (1.0 - Om0) * a ** (-3.0 * (1.0 + w0 + wa))
    return Om0 * a ** (-3.0) + de


def _Om(a, Om0, w0, wa):
    """Matter density fraction Omega_m(a)."""
    return Om0 * a ** (-3.0) / _H2(a, Om0, w0, wa)


def _w(a, w0, wa):
    return w0 + wa * (1.0 - a)


def _dlnD_dlna(a, Om0, w0, wa):
    """Reference growth rate f(a) = d ln D / d ln a from the ODE."""
    H2 = _H2(a, Om0, w0, wa)
    Om = _Om(a, Om0, w0, wa)
    w = _w(a, w0, wa)
    dH2 = -3.0 * Om0 * a ** (-4.0) - 3.0 * (1.0 + w) * (H2 - Om0 * a ** (-3.0))
    dH2dlna = a * dH2
    denom = 0.5 * Om - 1.5 * (1.0 + w) + 0.5 * dH2dlna / H2
    return 0.5 * (3.0 * Om - 1.0 - 3.0 * w) / denom


def _integrate_D(Om0, w0, wa):
    """Integrate the growth ODE from a=1 to a_min.  Returns (a, D, f)."""
    a_min = 0.05
    # state y = [ln a, ln D, f]; evolve t = ln a from 0 down to ln(a_min)
    def rhs(t, y):
        ln_a = y[0]
        a = math.exp(ln_a)
        f = y[2]
        H2 = _H2(a, Om0, w0, wa)
        Om = _Om(a, Om0, w0, wa)
        w = _w(a, w0, wa)
        dH2 = -3.0 * Om0 * a ** (-4.0) - 3.0 * (1.0 + w) * (H2 - Om0 * a ** (-3.0))
        dH2dlna = a * dH2
        denom = 0.5 * Om - 1.5 * (1.0 + w) + 0.5 * dH2dlna / H2
        f_target = 0.5 * (3.0 * Om - 1.0 - 3.0 * w) / denom
        # d f/d ln a = (f^2 - f*f_target - 0.5*f*dH2dlna/H2) / (f - f_target)
        df = (f * f - f * f_target - 0.5 * f * dH2dlna / H2) / (f - f_target)
        return [1.0, f, df]

    f0 = _dlnD_dlna(1.0, Om0, w0, wa)
    t_span = (0.0, math.log(a_min))
    sol = solve_ivp(rhs, t_span, [0.0, 0.0, f0], method="RK45",
                    rtol=1e-9, atol=1e-12, dense_output=True, max_step=0.02)
    a = np.exp(sol.t)
    D = np.exp(sol.y[1])
    f = sol.y[2]
    return a, D, f


def _D_norm(a_arr, f_arr):
    """Normalized growth factor D(a)/D(1) from the growth rate f(a).

    D(a)/D(1) = exp(+ integral_0^{ln a} f(exp(x)) dx)  (matches the
    evaluator's exp(-integral_ln(a)^0 ...), i.e. integral from 0 to ln a).
    """
    ln_a = np.log(a_arr)
    # cumulative integral of f from 0 to each ln_a (ln_a <= 0)
    integrand = f_arr
    x = ln_a
    # sort by x increasing (most negative first) so we can cumtrapz from a_min
    order = np.argsort(x)
    x_s = x[order]
    f_s = integrand[order]
    cum = np.zeros_like(x_s)
    cum[1:] = np.cumsum(0.5 * (f_s[1:] + f_s[:-1]) * np.diff(x_s))
    # integral from 0 to x: interpolate cum at x (cum is from x_min to x)
    # integral_0^x = integral_{x_min}^x - integral_{x_min}^0
    val = np.interp(x, x_s, cum)
    val0 = np.interp(0.0, x_s, cum)
    return np.exp(val - val0)


# ---------------------------------------------------------------------------
# Model (candidate) quantities
# ---------------------------------------------------------------------------

def _model_gamma(c, a, Om0, w0, wa):
    Om = _Om(a, Om0, w0, wa)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c[0] + c[1] * q + c[2] * q ** 2 + c[3] * u + c[4] * wa * a + c[5] * u * q


def _model_f(c, a, Om0, w0, wa):
    Om = _Om(a, Om0, w0, wa)
    g = _model_gamma(c, a, Om0, w0, wa)
    return Om ** g


def _model_D(c, a_arr, Om0, w0, wa):
    f = _model_f(c, a_arr, Om0, w0, wa)
    return _D_norm(a_arr, f)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _objective(c, models, grids):
    """100 * max relative error of f and normalized D, plus constraint penalty."""
    max_err = 0.0
    for (Om0, w0, wa), (a, f_ref, D_ref) in zip(models, grids):
        f_m = _model_f(c, a, Om0, w0, wa)
        D_m = _model_D(c, a, Om0, w0, wa)
        e_f = float(np.max(np.abs((f_m - f_ref) / f_ref)))
        e_D = float(np.max(np.abs((D_m - D_ref) / D_ref)))
        max_err = max(max_err, e_f, e_D)
    # constraint: gamma in [0.1, 1.2]
    gmin = 1.0
    gmax = 0.0
    for (Om0, w0, wa), (a, _, _) in zip(models, grids):
        g = _model_gamma(c, a, Om0, w0, wa)
        gmin = min(gmin, float(np.min(g)))
        gmax = max(gmax, float(np.max(g)))
    penalty = 0.0
    if gmin < 0.1:
        penalty += (0.1 - gmin) * 50.0
    if gmax > 1.2:
        penalty += (gmax - 1.2) * 50.0
    return 100.0 * max_err + penalty


def _objective_fast(c, models, grids):
    """Cheaper objective on a subsampled grid (for inner loop)."""
    max_err = 0.0
    for (Om0, w0, wa), (a, f_ref, D_ref) in zip(models, grids):
        idx = np.arange(0, len(a), 2)
        aa = a[idx]
        f_m = _model_f(c, aa, Om0, w0, wa)
        D_m = _model_D(c, aa, Om0, w0, wa)
        e_f = float(np.max(np.abs((f_m - f_ref[idx]) / f_ref[idx])))
        e_D = float(np.max(np.abs((D_m - D_ref[idx]) / D_ref[idx])))
        max_err = max(max_err, e_f, e_D)
    gmin, gmax = 1.0, 0.0
    for (Om0, w0, wa), (a, _, _) in zip(models, grids):
        g = _model_gamma(c, a, Om0, w0, wa)
        gmin = min(gmin, float(np.min(g)))
        gmax = max(gmax, float(np.max(g)))
    penalty = 0.0
    if gmin < 0.1:
        penalty += (0.1 - gmin) * 50.0
    if gmax > 1.2:
        penalty += (gmax - 1.2) * 50.0
    return 100.0 * max_err + penalty


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def _optimize(models, grids, deadline):
    from scipy.optimize import minimize, differential_evolution

    bounds = [(-10.0, 10.0)] * 6
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    best_x = x0.copy()
    best_val = _objective_fast(x0, models, grids)

    t0 = time.time()
    time_left = deadline - (t0 - t0) - 8.0  # reserve 8s safety margin

    # Stage 1: differential evolution (global) if time permits
    if time_left > 25.0:
        try:
            de = differential_evolution(
                lambda c: _objective_fast(c, models, grids),
                bounds, seed=12345, maxiter=40, popsize=12,
                tol=1e-6, mutation=(0.4, 1.0), recombination=0.8,
                polish=False, init="latinhypercube",
                update_interval=1,
            )
            if de.x is not None:
                v = _objective_fast(de.x, models, grids)
                if v < best_val:
                    best_val = v
                    best_x = de.x.copy()
        except Exception:
            pass

    t1 = time.time()
    time_left = deadline - (t1 - t0) - 8.0

    # Stage 2: local polish with L-BFGS-B (bounded)
    if time_left > 6.0:
        try:
            res = minimize(
                lambda c: _objective_fast(c, models, grids),
                best_x, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 400, "ftol": 1e-14, "gtol": 1e-10},
            )
            if res.x is not None:
                v = _objective_fast(res.x, models, grids)
                if v < best_val:
                    best_val = v
                    best_x = res.x.copy()
        except Exception:
            pass

    t2 = time.time()
    time_left = deadline - (t2 - t0) - 5.0

    # Stage 3: high-precision polish on the FULL objective
    if time_left > 4.0:
        try:
            res = minimize(
                lambda c: _objective(c, models, grids),
                best_x, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 200, "ftol": 1e-16, "gtol": 1e-12},
            )
            if res.x is not None:
                v = _objective(res.x, models, grids)
                if v < best_val:
                    best_val = v
                    best_x = res.x.copy()
        except Exception:
            pass

    return best_x


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def construct():
    deadline = 78.0  # wall-clock safety margin below the 90s kill
    t_start = time.time()

    # Validation-like CPL cosmologies (disjoint, spanning the parameter space)
    models = [
        (0.30, -1.0, 0.0),   # LCDM
        (0.30, -0.9, 0.0),
        (0.30, -1.1, 0.0),
        (0.30, -1.0, 0.5),
        (0.30, -1.0, -0.5),
        (0.25, -1.0, 0.0),
        (0.35, -1.0, 0.0),
        (0.30, -0.5, 0.0),
        (0.30, -1.5, 0.0),
        (0.30, -1.0, 1.0),
        (0.30, -1.0, -1.0),
        (0.32, -0.8, 0.3),
    ]

    grids = []
    for (Om0, w0, wa) in models:
        a, D, f = _integrate_D(Om0, w0, wa)
        Dn = _D_norm(a, f)
        grids.append((a, f, Dn))

    if time.time() - t_start > deadline:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    best_x = _optimize(models, grids, deadline)

    # Clamp to valid range
    best_x = np.clip(best_x, -10.0, 10.0)
    return [float(x) for x in best_x]
# EVOLVE-BLOCK-END
