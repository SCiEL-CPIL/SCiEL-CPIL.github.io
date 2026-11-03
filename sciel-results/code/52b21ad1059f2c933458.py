import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

_START = time.time()
_DEADLINE = 170.0  # safety margin under 210s

def _M_table():
    M = np.zeros(15)
    M[0] = np.sqrt(np.pi)
    for k in range(1, 15):
        M[k] = (2*k - 1) / 2.0 * M[k-1]
    return M
_M = _M_table()

def _bound(c0, c4, c8):
    c12 = -(c0 + 3.0*c4 + 105.0*c8) / 10395.0
    a = np.zeros(7)
    a[1] = -6.0*c4 - 420.0*c8 - 332640.0*c12
    a[2] = c4 + 210.0*c8 + 151200.0*c12
    a[3] = -28.0*c8 - 30240.0*c12
    a[4] = c8 + 2520.0*c12
    a[5] = -84.0*c12
    a[6] = c12
    I2 = 0.0; Ix2 = 0.0
    for k in range(1,7):
        for l in range(1,7):
            I2 += a[k]*a[l]*_M[k+l]
            Ix2 += a[k]*a[l]*_M[k+l+1]
    b = np.zeros(8)
    for m in range(0,7):
        b[m] = 2.0*(m+1)*a[m+1] - a[m]
    If2 = 0.0
    for m in range(0,7):
        for n in range(0,7):
            If2 += b[m]*b[n]*_M[m+n+1]
    if not (np.isfinite(I2) and np.isfinite(Ix2) and np.isfinite(If2)):
        return np.inf
    if I2 <= 1e-30 or Ix2 <= 1e-30 or If2 <= 1e-30:
        return np.inf
    return np.sqrt(Ix2*If2)/I2

def _obj(x):
    return _bound(x[0], x[1], x[2])

def construct():
    global _START
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best = seed.copy()
    best_v = _bound(*seed)
    if not np.isfinite(best_v):
        return seed
    def time_ok():
        return (time.time() - _START) < _DEADLINE
    # local polish from seed
    try:
        if time_ok():
            res = minimize(_obj, seed, method='L-BFGS-B',
                           bounds=[(-3,3),(-0.5,0.5),(-0.05,0.05)],
                           options={'maxiter': 4000, 'maxfun': 40000})
            if res.success and np.isfinite(res.fun) and res.fun < best_v:
                best, best_v = res.x, res.fun
    except Exception:
        pass
    # multi-start Nelder-Mead
    rng = np.random.default_rng(0)
    base = np.array([-1.13, 0.04, 0.000308642])
    for _ in range(60):
        if not time_ok():
            break
        x0 = base + rng.normal(0, 0.15, size=3)
        x0 = np.clip(x0, [-3,-0.5,-0.05], [3,0.5,0.05])
        try:
            res = minimize(_obj, x0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol':1e-10,'fatol':1e-12})
            if np.isfinite(res.fun) and res.fun < best_v:
                best, best_v = res.x, res.fun
                # polish
                res2 = minimize(_obj, best, method='L-BFGS-B',
                                bounds=[(-3,3),(-0.5,0.5),(-0.05,0.05)],
                                options={'maxiter':2000,'maxfun':20000})
                if np.isfinite(res2.fun) and res2.fun < best_v:
                    best, best_v = res2.x, res2.fun
        except Exception:
            pass
    if not np.isfinite(best_v) or best_v > _bound(*seed):
        return seed
    return np.array([float(best[0]), float(best[1]), float(best[2])])
