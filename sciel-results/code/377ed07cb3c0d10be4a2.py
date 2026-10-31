"""Reformulated minimization with analytic gradient + multi-start L-BFGS-B."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 180.0
_MARGIN = 20.0          # safety margin before the 210 s kill


def _a12(a0, a4, a8):
    # g(0) = 0  =>  a12 = -(3*a0 + 10*a4 + a8)/105
    return -(3.0 * a0 + 10.0 * a4 + a8) / 105.0


def _norm_sq(a0, a4, a8):
    a12 = _a12(a0, a4, a8)
    return (1.0 * a0 ** 2 + 24.0 * a4 ** 2 + 2016.0 * a8 ** 2
            + 129024.0 * a12 ** 2
            + 48.0 * a0 * a4 + 4032.0 * a0 * a8
            + 258048.0 * a0 * a12
            + 48384.0 * a4 * a8 + 3091200.0 * a4 * a12
            + 1290240.0 * a8 * a12)


def _mean_sq(a0, a4, a8):
    a12 = _a12(a0, a4, a8)
    return (1.0 * a0 ** 2 + 12.0 * a4 ** 2 + 1344.0 * a8 ** 2
            + 870912.0 * a12 ** 2
            + 4.0 * a0 * a4 + 2688.0 * a0 * a8
            + 1741824.0 * a0 * a12
            + 2688.0 * a4 * a8 + 2060800.0 * a4 * a12
            + 1030400.0 * a8 * a12)


def _x2(a0, a4, a8):
    a12 = _a12(a0, a4, a8)
    return (1.0 * a0 ** 2 + 30.0 * a4 ** 2 + 10080.0 * a8 ** 2
            + 15896320.0 * a12 ** 2
            + 8.0 * a0 * a4 + 20160.0 * a0 * a8
            + 31792640.0 * a0 * a12
            + 20160.0 * a4 * a8 + 37094400.0 * a4 * a12
            + 18547200.0 * a8 * a12)


def _x4(a0, a4, a8):
    a12 = _a12(a0, a4, a8)
    return (1.0 * a0 ** 2 + 50.0 * a4 ** 2 + 80640.0 * a8 ** 2
            + 397408000.0 * a12 ** 2
            + 12.0 * a0 * a4 + 161280.0 * a0 * a8
            + 794816000.0 * a0 * a12
            + 161280.0 * a4 * a8 + 927360000.0 * a4 * a12
            + 463680000.0 * a8 * a12)


def _fval(a):
    a0, a4, a8 = a
    N = _norm_sq(a0, a4, a8)
    if N <= 0.0:
        return 1e12
    m2 = _mean_sq(a0, a4, a8) / N
    x2 = _x2(a0, a4, a8) / N
    x4 = _x4(a0, a4, a8) / N
    var_x = x2 - m2
    if var_x <= 0.0:
        return 1e12
    var_xi = x4 / 4.0
    return np.sqrt(var_x) * np.sqrt(var_xi)


def _fgrad(a):
    a0, a4, a8 = a
    N = _norm_sq(a0, a4, a8)
    if N <= 0.0:
        return np.array([1e6, 1e6, 1e6])
    dN = np.array([2.0 * a0 + 48.0 * a4 + 4032.0 * a8
                   - 6.0 / 105.0 * (1.0 + 20.0 * a4 / a0 + a8 / a0) * 0  # placeholder
                   , 0.0, 0.0])
    # Compute dN, dm2, dx2, dx4 analytically (a12 depends on a0,a4,a8).
    a12 = _a12(a0, a4, a8)
    da12 = np.array([-3.0 / 105.0, -10.0 / 105.0, -1.0 / 105.0])

    # dN/da
    dN = (np.array([2.0 * a0, 48.0 * a4, 2016.0 * a8])
          + 2.0 * 129024.0 * a12 * da12
          + np.array([48.0 * a4 + 4032.0 * a8 + 258048.0 * a12,
                      48.0 * a0 + 48384.0 * a8 + 3091200.0 * a12,
                      4032.0 * a0 + 3091200.0 * a4 + 2580480.0 * a12]))
    dm2 = (np.array([2.0 * a0, 24.0 * a4, 1344.0 * a8])
           + 2.0 * 870912.0 * a12 * da12
           + np.array([4.0 * a4 + 2688.0 * a8 + 1741824.0 * a12,
                       4.0 * a0 + 2688.0 * a8 + 2060800.0 * a12,
                       2688.0 * a0 + 2060800.0 * a4 + 2060800.0 * a12]))
    dx2 = (np.array([2.0 * a0, 60.0 * a4, 10080.0 * a8])
           + 2.0 * 15896320.0 * a12 * da12
           + np.array([8.0 * a4 + 20160.0 * a8 + 31792640.0 * a12,
                       8.0 * a0 + 20160.0 * a8 + 37094400.0 * a12,
                       20160.0 * a0 + 37094400.0 * a4 + 37094400.0 * a12]))
    dx4 = (np.array([2.0 * a0, 100.0 * a4, 80640.0 * a8])
           + 2.0 * 397408000.0 * a12 * da12
           + np.array([12.0 * a4 + 161280.0 * a8 + 794816000.0 * a12,
                       12.0 * a0 + 161280.0 * a8 + 927360000.0 * a12,
                       161280.0 * a0 + 927360000.0 * a4 + 927360000.0 * a12]))

    m2 = _mean_sq(a0, a4, a8) / N
    x2 = _x2(a0, a4, a8) / N
    x4 = _x4(a0, a4, a8) / N
    var_x = x2 - m2
    if var_x <= 0.0:
        return np.array([1e6, 1e6, 1e6])
    var_xi = x4 / 4.0

    dm2d = dm2 / N - m2 * dN / N
    dx2d = dx2 / N - x2 * dN / N
    dx4d = dx4 / N - x4 * dN / N

    dvarx = dx2d - dm2d
    dvarxi = dx4d / 4.0

    # d(sqrt(var_x)*sqrt(var_xi)) = (var_xi/(2 sqrt(var_x))) dvarx
    #                            + (var_x/(2 sqrt(var_xi))) dvarxi
    grad = (var_xi / (2.0 * np.sqrt(var_x))) * dvarx \
         + (var_x / (2.0 * np.sqrt(var_xi))) * dvarxi
    return grad


def construct():
    deadline = _START + _DEADLINE - _MARGIN
    best = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _fval(best)

    rng = np.random.default_rng(12345)
    base = best.copy()
    starts = [base.copy()]
    # deterministic multi-start around the prior + random perturbations
    for r in [0.05, 0.1, 0.2, 0.4, 0.8, 1.5]:
        starts.append(base + r * rng.standard_normal(3))
    for _ in range(40):
        starts.append(base + 1.5 * rng.standard_normal(3))
    # a couple of broad exploratory starts
    for _ in range(20):
        starts.append(2.0 * rng.standard_normal(3))

    bounds = [(-10.0, 10.0)] * 3

    for s in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(_fval, s, jac=_fgrad, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.x is not None:
                v = _fval(res.x)
                if np.isfinite(v) and v < best_val:
                    best_val = v
                    best = res.x.copy()
        except Exception:
            continue

    return np.asarray(best, dtype=float)
# EVOLVE-BLOCK-END
