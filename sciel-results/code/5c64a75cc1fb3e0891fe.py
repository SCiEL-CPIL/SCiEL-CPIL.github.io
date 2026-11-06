"""Evolved program: fit the six-coefficient growth-index ansatz to a built-in
GR ODE reference over a validation grid, minimizing worst relative error."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 85.0  # safety margin under the 90 s wall-clock limit
_t_start = time.time()


def _om_of_a(a, Om0, w0, wa):
    """Matter density parameter Om(a) for smooth CPL dark energy."""
    a = np.asarray(a, dtype=float)
    de = 1.0 - Om0
    if abs(wa) < 1e-12:
        e2 = Om0 * a ** (-3.0) + de * a ** (-3.0 * (1.0 + w0))
    else:
        c = 1.0 + w0 + wa
        e2 = Om0 * a ** (-3.0) + de * a ** (-3.0 * c) * np.exp(3.0 * wa * (1.0 - a))
    return Om0 * a ** 3.0 / e2


def _f_of_a(a, Om0, w0, wa, c):
    c0, c1, c2, c3, c4, c5 = c
    om = _om_of_a(a, Om0, w0, wa)
    q = 1.0 - om
    u = 1.0 + w0 + wa * (1.0 - a)
    g = c0 + c1 * q + c2 * q * q + c3 * u + c4 * wa * a + c5 * u * q
    g = np.clip(g, 0.1, 1.2)
    return om ** g


def _d_norm(a, Om0, w0, wa, c):
    """Normalized growth factor D(a)/D(1) = exp(-int_{ln a}^0 f(e^x) dx)."""
    a = np.asarray(a, dtype=float)
    n = 400
    xs = np.linspace(0.0, 1.0, n)
    ax = np.exp(-xs)
    f = _f_of_a(ax, Om0, w0, wa, c)
    # cumulative integral from x=0 (a=1) to x (a=e^-x)
    cum = np.zeros_like(xs)
    if n > 1:
        cum[1:] = np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(xs))
    lnD = -cum
    # interpolate lnD(a)
    return np.exp(np.interp(np.log(a), -xs, lnD))


def _rel_err(a, Om0, w0, wa, c):
    om = _om_of_a(a, Om0, w0, wa)
    q = 1.0 - om
    u = 1.0 + w0 + wa * (1.0 - a)
    g = c[0] + c[1] * q + c[2] * q * q + c[3] * u + c[4] * wa * a + c[5] * u * q
    g = np.clip(g, 0.1, 1.2)
    f = om ** g

    # reference f via ODE
    n = 300
    agrid = np.linspace(0.05, 1.0, n)
    xg = -np.log(agrid)
    omg = _om_of_a(agrid, Om0, w0, wa)
    H2 = omg + (1.0 - omg) * np.exp(3.0 * (1.0 + w0 + wa * (1.0 - agrid)))
    omg2 = np.gradient(omg, xg)
    f_ref = (3.0 / 2.0) * omg + (1.0 / 2.0) * omg2

    # interpolate reference f at evaluation a
    f_ref_at = np.interp(np.log(a), -xg, f_ref)

    # reference D via ODE integration
    nD = 300
    aD = np.linspace(0.05, 1.0, nD)
    xD = -np.log(aD)
    oD = _om_of_a(aD, Om0, w0, wa)
    fD = (3.0 / 2.0) * oD + (1.0 / 2.0) * np.gradient(oD, xD)
    cumD = np.zeros_like(xD)
    if nD > 1:
        cumD[1:] = np.cumsum(0.5 * (fD[1:] + fD[:-1]) * np.diff(xD))
    lnD_ref = -cumD
    d_ref = np.exp(np.interp(np.log(a), -xD, lnD_ref))

    d = _d_norm(a, Om0, w0, wa, c)

    ef = np.abs(f - f_ref_at) / np.maximum(np.abs(f_ref_at), 1e-12)
    ed = np.abs(d - d_ref) / np.maximum(np.abs(d_ref), 1e-12)
    return max(ef.max(), ed.max())


def _build_problem():
    """Build internal validation grid: cosmologies x scale factors."""
    cosmo = [
        (0.27, -1.0, 0.0),    # LCDM
        (0.30, -1.0, 0.0),
        (0.25, -1.0, 0.0),
        (0.30, -0.9, -0.3),   # CPL
        (0.27, -0.8, -0.6),
        (0.30, -1.0, -0.5),
        (0.25, -0.95, 0.1),
        (0.32, -1.1, -0.2),
    ]
    a_vals = np.concatenate([
        np.linspace(0.05, 0.20, 12),
        np.linspace(0.20, 0.60, 12),
        np.linspace(0.60, 1.0, 12),
    ])
    return cosmo, a_vals


def construct():
    global _t_start
    _t_start = time.time()
    cosmo, a_vals = _build_problem()
    bounds = [(-10.0, 10.0)] * 6

    def objective(c):
        worst = 0.0
        for (Om0, w0, wa) in cosmo:
            e = _rel_err(a_vals, Om0, w0, wa, c)
            if e > worst:
                worst = e
        return worst

    from scipy.optimize import minimize

    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_val = objective(best_c)

    starts = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.1, -0.1, 0.0, 0.0, 0.0],
        [0.6, -0.1, 0.05, 0.0, 0.0, 0.0],
        [0.5, 0.05, 0.0, 0.05, 0.0, -0.02],
        [0.55, -0.05, 0.0, 0.0, 0.05, 0.0],
        [0.6, 0.0, -0.05, -0.05, 0.0, 0.03],
    ]

    for s in starts:
        if time.time() - _t_start > _DEADLINE:
            break
        try:
            res = minimize(objective, np.array(s, dtype=float),
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-7})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x
        except Exception:
            pass

    # final polish
    if time.time() - _t_start < _DEADLINE:
        try:
            res = minimize(objective, best_c, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-11, 'gtol': 1e-9})
            if res.fun < best_val:
                best_c = res.x
        except Exception:
            pass

    return [float(x) for x in best_c]
# EVOLVE-BLOCK-END
