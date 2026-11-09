"""Compact six-coefficient growth-index ansatz fit against GR ODE reference.

The ansatz:
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    f = Om(a)^gamma
    D(a)/D(1) = exp(-integral_{ln a}^{0} f(x) dx)

We minimize the worst relative error of both f and normalized D over a set of
representative CPL cosmologies and a dense grid of scale factors, using the
GR second-order ODE as the independent reference.
"""
import math
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

_DEADLINE = time.time() + 84.0
_MAX_COEFF = 10.0


def _remaining():
    return _DEADLINE - time.time()


def _Om(a, Om0, w0, wa):
    y = a ** (-3.0)
    de = a ** (-3.0 * (1.0 + w0 + wa))
    return Om0 * y / (Om0 * y + de)


def _H2(a, Om0, w0, wa):
    y = a ** (-3.0)
    de = a ** (-3.0 * (1.0 + w0 + wa))
    return Om0 * y + de


def _rhs(ln_a, y, Om0, w0, wa):
    a = math.exp(ln_a)
    H2 = _H2(a, Om0, w0, wa)
    dH2 = (-3.0 * a ** (-4.0) * (Om0 + (1.0 + w0 + wa) * a ** (-3.0 * (1.0 + w0 + wa))))
    Om = _Om(a, Om0, w0, wa)
    d1 = (3.0 + 0.5 * dH2 / H2) * y[1] - 1.5 * Om * y[0]
    return [y[1], d1]


def _solve_reference(a_grid, Om0, w0, wa):
    ln_a = np.log(a_grid)
    sol = solve_ivp(_rhs, (ln_a[0], ln_a[-1]), [1.0, 0.0],
                    args=(Om0, w0, wa), method="RK45",
                    rtol=1e-10, atol=1e-13, dense_output=False)
    if not sol.success:
        return None
    D = sol.y[0]
    D = D / D[-1]
    f = sol.y[1] / D
    return D, f


_COSMO = [(0.3, -1.0, 0.0), (0.3, -1.0, 0.3), (0.3, -1.0, -0.5),
          (0.27, -1.0, 0.0), (0.3, -0.9, 0.2), (0.3, -1.1, 0.4),
          (0.3, -0.8, 0.5), (0.25, -1.0, 0.0)]
_A_GRID = np.linspace(0.01, 1.0, 300)
_LN_A = np.log(_A_GRID)

# Pre-compute reference solutions and auxiliary arrays
_REF_CACHE = []
for (Om0, w0, wa) in _COSMO:
    ref = _solve_reference(_A_GRID, Om0, w0, wa)
    if ref is None:
        _REF_CACHE.append(None)
        continue
    D_ref, f_ref = ref
    Om_arr = np.array([_Om(a, Om0, w0, wa) for a in _A_GRID])
    q_arr = 1.0 - Om_arr
    u_arr = 1.0 + w0 + wa * (1.0 - _A_GRID)
    _REF_CACHE.append((D_ref, f_ref, Om_arr, q_arr, u_arr, wa))


def _objective(c):
    c = np.clip(c, -_MAX_COEFF, _MAX_COEFF)
    worst = 0.0
    for entry in _REF_CACHE:
        if entry is None:
            continue
        D_ref, f_ref, Om_arr, q_arr, u_arr, wa = entry
        g = (c[0] + c[1] * q_arr + c[2] * q_arr * q_arr + c[3] * u_arr
             + c[4] * wa * _A_GRID + c[5] * u_arr * q_arr)
        f_cand = Om_arr ** g
        # D_cand via cumulative trapezoid from a=1 backward
        dx = np.diff(_LN_A)
        trap = 0.5 * (f_cand[:-1] + f_cand[1:]) * dx
        cum = np.zeros(len(_LN_A))
        cum[:-1] = np.cumsum(trap[::-1])[::-1]
        D_cand = np.exp(-cum)
        rel_f = np.abs(f_cand - f_ref) / np.abs(f_ref)
        rel_D = np.abs(D_cand - D_ref) / np.abs(D_ref)
        e = float(np.max(np.maximum(rel_f, rel_D)))
        if e > worst:
            worst = e
    return worst


def construct():
    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_val = float("inf")

    starts = [
        np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.5, 0.1, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.6, -0.1, 0.05, 0.0, 0.0, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.05, 0.0, 0.0]),
        np.array([0.5, 0.05, 0.0, 0.05, 0.0, 0.0]),
        np.array([0.55, 0.05, -0.02, 0.02, 0.0, 0.0]),
        np.array([0.52, 0.08, 0.0, -0.03, 0.01, 0.0]),
        np.array([0.58, -0.05, 0.03, 0.03, -0.01, 0.02]),
    ]

    try:
        for s in starts:
            if _remaining() < 10.0:
                break
            res = minimize(_objective, s, method="Nelder-Mead",
                           options={"maxiter": 1000, "xatol": 1e-7, "fatol": 1e-8,
                                    "adaptive": True})
            if res.fun < best_val:
                cand = np.clip(res.x, -_MAX_COEFF, _MAX_COEFF)
                v = _objective(cand)
                if v < best_val:
                    best_val = v
                    best_c = cand
    except Exception:
        pass

    # Final polish with tighter tolerances if time permits
    if _remaining() > 5.0:
        try:
            res = minimize(_objective, best_c, method="Nelder-Mead",
                           options={"maxiter": 500, "xatol": 1e-9, "fatol": 1e-10,
                                    "adaptive": True})
            if res.fun < best_val:
                cand = np.clip(res.x, -_MAX_COEFF, _MAX_COEFF)
                v = _objective(cand)
                if v < best_val:
                    best_val = v
                    best_c = cand
        except Exception:
            pass

    out = [float(min(max(x, -_MAX_COEFF), _MAX_COEFF)) for x in best_c]
    return out
