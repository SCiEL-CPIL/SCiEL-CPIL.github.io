import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 630.0
_MARGIN = 15.0

def _time_left():
    return _DEADLINE - (time.time() - _START) - _MARGIN

def _logspace(a, b, n):
    if n == 1:
        return np.array([10 ** ((a + b) / 2.0)])
    return 10.0 ** np.linspace(a, b, n)

def construct():
    m = 12
    n = 1
    N = 2 * m  # 24

    def _coeffs(x):
        # product_{k=1}^{m} (L_n(x) - L_n(x_k))^2 as a coefficient vector (ascending)
        ln = np.polynomial.laguerre.lagval(x, np.polynomial.laguerre.Laguerre.basis(n))
        c = np.array([1.0])
        for k in range(1, m + 1):
            d = ln - np.polynomial.laguerre.lagval(x[k - 1], np.polynomial.laguerre.Laguerre.basis(n))
            q = np.polynomial.polynomial.polymul(c, np.array([d, d]))
            c = q
        return c

    def _bound(x):
        c = _coeffs(x)
        P = np.polynomial.Polynomial(c)
        P2 = P * P
        w = np.polynomial.Polynomial([1.0, -1.0])  # e^{-x} = 1 - x
        PW2 = P2 * w
        # integral over [0, inf) via generalized Gauss-Laguerre quadrature
        nodes, weights = np.polynomial.laguerre.laggauss(80)
        PW2v = PW2(nodes)
        I = np.sum(weights * PW2v)
        if I <= 0 or not np.isfinite(I):
            return np.inf
        return 1.0 / np.sqrt(I)

    # seed: uniform positive spacing (the reference)
    seed = 4.0 * np.arange(1, m + 1, dtype=float)
    best = seed.copy()
    best_v = _bound(best)

    # multi-start: several initial spreads/scales, then Nelder-Mead polish
    def _start(scale, spread):
        # increasing positive locations, roughly log-ish
        base = np.linspace(0.5, 1.0, m)
        return scale * np.power(base, spread)

    starts = []
    for scale in (1.0, 2.0, 4.0, 8.0, 16.0):
        for spread in (0.5, 1.0, 1.5, 2.0):
            starts.append(_start(scale, spread))
    starts.append(seed.copy())

    for s in starts:
        if _time_left() <= 5.0:
            break
        try:
            res = minimize(_bound, s, method="Nelder-Mead",
                           options={"maxiter": 400, "maxfev": 1200,
                                    "xatol": 1e-5, "fatol": 1e-7})
            if res.success is False and not np.isfinite(res.fun):
                continue
            x = np.asarray(res.x, dtype=float)
            if np.any(x <= 0) or np.any(~np.isfinite(x)):
                continue
            v = float(res.fun)
            if np.isfinite(v) and v < best_v:
                best_v = v
                best = x.copy()
        except Exception:
            continue

    # final polish on the best found
    if _time_left() > 30.0:
        try:
            res = minimize(_bound, best, method="Nelder-Mead",
                           options={"maxiter": 800, "maxfev": 2500,
                                    "xatol": 1e-7, "fatol": 1e-9})
            x = np.asarray(res.x, dtype=float)
            if np.all(x > 0) and np.all(np.isfinite(x)) and np.isfinite(res.fun):
                if float(res.fun) < best_v:
                    best_v = float(res.fun)
                    best = x.copy()
        except Exception:
            pass

    return [float(v) for v in best]
