import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import minimize, minimize_scalar
import time

_T0 = time.time()
_KILL = 210.0
_MARGIN = 30.0
_DEADLINE = _KILL - _MARGIN

def _time_left():
    return _DEADLINE - (time.time() - _T0)

_NQH = 160
_X, _W = np.polynomial.hermite.hermgauss(_NQH)
_H0 = eval_hermite(0, _X)
_H4 = eval_hermite(4, _X)
_H8 = eval_hermite(8, _X)
_H12 = eval_hermite(12, _X)
_X4 = _X * _X * _X * _X

# seed ratios
_SEED_A = -113.0/100.0
_SEED_B = 1.0/25.0
_SEED_C = 1.0/3240.0
_SEED_R = _SEED_B/_SEED_A
_SEED_S = _SEED_C/_SEED_A

def _moments(r, s):
    t = -(1.0 + 12.0*r + 1680.0*s)/665280.0
    poly = _H0 + r*_H4 + s*_H8 + t*_H12
    p2 = poly*poly
    m0 = float(np.dot(_W, p2))
    m4 = float(np.dot(_W, _X4*p2))
    return m0, m4

def _obj(v):
    r, s = v
    m0, m4 = _moments(r, s)
    if m0 <= 1e-30 or not np.isfinite(m0):
        return 1e30
    val = m4/(m0*m0)
    if not np.isfinite(val):
        return 1e30
    return val

def _eval1(r, s):
    return _obj((r, s))

def construct():
    best_r, best_s = _SEED_R, _SEED_S
    best_val = _eval1(best_r, best_s)
    if not np.isfinite(best_val):
        best_val = 1e30

    rng = np.random.default_rng(12345)

    # 1) coarse grid
    if _time_left() > 5:
        gr = np.linspace(-4.0, 4.0, 80)
        gs = np.linspace(-0.4, 0.4, 80)
        starts = []
        for r in gr:
            for s in gs:
                v = _eval1(r, s)
                if v < best_val:
                    best_val = v; best_r, best_s = r, s
                starts.append((r, s, v))
        starts.sort(key=lambda x: x[2])
        top = [x[:2] for x in starts[:40]]
    else:
        top = [(best_r, best_s)]

    # 2) multi-start local (BFGS + Nelder-Mead)
    extra = [(float(rng.uniform(-4,4)), float(rng.uniform(-0.4,0.4))) for _ in range(300)]
    all_starts = top + extra + [(best_r, best_s)]
    for (r0, s0) in all_starts:
        if _time_left() < 15:
            break
        for method in ('BFGS', 'Nelder-Mead'):
            try:
                res = minimize(_obj, (r0, s0), method=method,
                               options={'maxiter': 400, 'maxfev': 800})
                if res.success and np.isfinite(res.fun) and res.fun < best_val:
                    best_val = float(res.fun)
                    best_r, best_s = float(res.x[0]), float(res.x[1])
            except Exception:
                pass

    # 3) 1D Brent slices
    if _time_left() > 20:
        for s in np.linspace(-0.4, 0.4, 120):
            if _time_left() < 15: break
            try:
                rs = minimize_scalar(lambda r: _eval1(r, s), bounds=(-4.0, 4.0), method='bounded')
                if rs.success and np.isfinite(rs.fun) and rs.fun < best_val:
                    best_val = float(rs.fun); best_r, best_s = float(rs.x), float(s)
            except Exception:
                pass
        for r in np.linspace(-4.0, 4.0, 120):
            if _time_left() < 15: break
            try:
                rs = minimize_scalar(lambda s: _eval1(r, s), bounds=(-0.4, 0.4), method='bounded')
                if rs.success and np.isfinite(rs.fun) and rs.fun < best_val:
                    best_val = float(rs.fun); best_r, best_s = float(r), float(rs.x)
            except Exception:
                pass

    # 4) polish
    if _time_left() > 10:
        try:
            res = minimize(_obj, (best_r, best_s), method='Nelder-Mead',
                           options={'xatol':1e-12,'fatol':1e-14,'maxiter':5000,'maxfev':20000,'adaptive':True})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = float(res.fun); best_r, best_s = float(res.x[0]), float(res.x[1])
        except Exception:
            pass

    # return with a=1 scale (scale invariance)
    return np.array([1.0, best_r, best_s])
