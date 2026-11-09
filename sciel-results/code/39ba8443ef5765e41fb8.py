"""Compact dark-energy growth-index fit: derive six coefficients, then polish."""
import math
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.integrate import quad

_DEADLINE = 80.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _budget_left(frac=0.9):
    return _DEADLINE * frac - _elapsed()


def _om(a, om0):
    return om0 * a ** (-3) / (om0 * a ** (-3) + 1.0 - om0)


def _u(a, w0, wa):
    return 1.0 + w0 + wa * (1.0 - a)


def _q(a, om0):
    return 1.0 - _om(a, om0)


def _gamma_ref(a, om0, w0, wa):
    return (5.0 / 7.0) * (_om(a, om0) / _u(a, w0, wa)) ** 0.38


def _gamma_6(a, om0, w0, wa, c):
    c0, c1, c2, c3, c4, c5 = c
    q = _q(a, om0)
    u = _u(a, w0, wa)
    return c0 + c1 * q + c2 * q * q + c3 * u + c4 * wa * a + c5 * u * q


def _f(a, om0, w0, wa, c):
    g = _gamma_6(a, om0, w0, wa, c)
    g = min(1.2, max(0.1, g))
    return _om(a, om0) ** g


def _f_ref(a, om0, w0, wa):
    g = _gamma_ref(a, om0, w0, wa)
    return _om(a, om0) ** g


def _D(a, om0, w0, wa, c):
    if a <= 1.0:
        val, _ = quad(lambda x: _f(math.exp(x), om0, w0, wa, c), math.log(a), 0.0)
        return math.exp(-val)
    val, _ = quad(lambda x: _f(math.exp(x), om0, w0, wa, c), 0.0, math.log(a))
    return math.exp(val)


def _D_ref(a, om0, w0, wa):
    if a <= 1.0:
        val, _ = quad(lambda x: _f_ref(math.exp(x), om0, w0, wa), math.log(a), 0.0)
        return math.exp(-val)
    val, _ = quad(lambda x: _f_ref(math.exp(x), om0, w0, wa), 0.0, math.log(a))
    return math.exp(val)


def _objective(c):
    c = np.clip(c, -10.0, 10.0)
    worst = 0.0
    for (om0, w0, wa) in _GRID:
        for a in _AVALS:
            g = _gamma_6(a, om0, w0, wa, c)
            if g < 0.1 - 1e-9 or g > 1.2 + 1e-9:
                return 1e6
            fr = _f_ref(a, om0, w0, wa)
            f = _f(a, om0, w0, wa, c)
            if fr > 0:
                worst = max(worst, abs(f - fr) / fr)
            Dr = _D_ref(a, om0, w0, wa)
            D = _D(a, om0, w0, wa, c)
            if Dr > 0:
                worst = max(worst, abs(D - Dr) / Dr)
    return 100.0 * worst


def _build_grid():
    om0 = np.array([0.20, 0.25, 0.30, 0.35, 0.40])
    w0 = np.array([-1.2, -1.0, -0.8, -0.5, 0.0])
    wa = np.array([-0.5, -0.3, 0.0, 0.3, 0.5])
    grid = []
    for o in om0:
        for w in w0:
            for wa_ in wa:
                grid.append((float(o), float(w), float(wa_)))
    return grid


def _build_aval():
    return np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])


_GRID = _build_grid()
_AVALS = _build_aval()


def _fit_least_squares():
    # Fit c to gamma_ref across grid and scale factors
    rows = []
    targets = []
    for (om0, w0, wa) in _GRID:
        for a in _AVALS:
            q = _q(a, om0)
            u = _u(a, w0, wa)
            row = [1.0, q, q * q, u, wa * a, u * q]
            rows.append(row)
            targets.append(_gamma_ref(a, om0, w0, wa))
    A = np.array(rows)
    b = np.array(targets)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.clip(sol, -10.0, 10.0)


def construct():
    global _GRID, _AVALS
    if _GRID is None:
        _GRID = _build_grid()
        _AVALS = _build_aval()

    # 1. Least-squares seed
    c0 = _fit_least_squares()
    best = c0.copy()
    best_val = _objective(best)

    if _budget_left() < 2.0:
        return [float(x) for x in best]

    # 2. Local polish: Nelder-Mead
    try:
        res = minimize(_objective, best, method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7,
                                'adaptive': True})
        if res.fun < best_val and _budget_left() > 1.0:
            best = np.clip(res.x, -10.0, 10.0)
            best_val = float(res.fun)
    except Exception:
        pass

    if _budget_left() < 2.0:
        return [float(x) for x in best]

    # 3. Gradient polish (L-BFGS-B) for a smoother local pass
    try:
        def _grad(c):
            h = 1e-5
            g = np.zeros(6)
            for i in range(6):
                cp = c.copy(); cp[i] += h
                cm = c.copy(); cm[i] -= h
                g[i] = (_objective(cp) - _objective(cm)) / (2 * h)
            return g
        res2 = minimize(_objective, best, jac=_grad, method='L-BFGS-B',
                        options={'maxiter': 200, 'gtol': 1e-6})
        if res2.fun < best_val and _budget_left() > 1.0:
            best = np.clip(res2.x, -10.0, 10.0)
            best_val = float(res2.fun)
    except Exception:
        pass

    # 4. Fine Nelder-Mead polish
    if _budget_left() > 3.0:
        try:
            res3 = minimize(_objective, best, method='Nelder-Mead',
                            options={'maxiter': 400, 'xatol': 1e-6, 'fatol': 1e-8,
                                     'adaptive': True, 'initial_simplex': 1e-3})
            if res3.fun < best_val:
                best = np.clip(res3.x, -10.0, 10.0)
                best_val = float(res3.fun)
        except Exception:
            pass

    best = np.clip(best, -10.0, 10.0)
    return [float(x) for x in best]
