import numpy as np
from scipy.optimize import minimize
import time

# Physicists' Hermite polynomials H0, H4, H8, H12 (descending powers), via H_{n+1}=2x H_n - 2n H_{n-1}
H0c = np.array([1.0])
H4c = np.array([16.0, 0.0, -48.0, 0.0, 12.0])
H8c = np.array([256.0, 0.0, -3584.0, 0.0, 13440.0, 0.0, -13440.0, 0.0, 1680.0])
H12c = np.array([4096.0, 0.0, -135168.0, 0.0, 1520640.0, 0.0, -7096320.0, 0.0,
                 13305600.0, 0.0, -7983360.0, 0.0, 665280.0])
H0_0 = H0c[-1]
H4_0 = H4c[-1]
H8_0 = H8c[-1]
H12_0 = H12c[-1]
_N = len(H12c)


def _pad(a):
    b = np.zeros(_N)
    b[:len(a)] = a
    return b


_P0 = _pad(H0c)
_P4 = _pad(H4c)
_P8 = _pad(H8c)
_P12 = _pad(H12c)


def _Pcoeffs(c1, c2):
    # c0 fixed to 1 (scale invariance). c3 enforces g(0)=0.
    c0 = 1.0
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    g = c0 * _P0 + c1 * _P4 + c2 * _P8 + c3 * _P12
    q = g[:-2]          # divide even g by x^2 -> degree-10 even polynomial
    return q[0::2]      # P(z) with z = x^2, degree 5, descending


def _largest(P):
    rts = np.roots(P)
    best = -1.0
    for z in rts:
        if abs(z.imag) > 1e-8 * max(1.0, abs(z.real)):
            continue
        zr = z.real
        if zr <= 1e-12:
            continue
        eps = 1e-6 * max(1.0, abs(zr))
        f1 = np.polyval(P, zr - eps)
        f2 = np.polyval(P, zr + eps)
        if f1 * f2 < 0 and zr > best:   # sign change => odd multiplicity root
            best = zr
    return best


def _obj(v):
    try:
        P = _Pcoeffs(v[0], v[1])
        b = _largest(P)
    except Exception:
        return 1e6
    if b < 0 or not np.isfinite(b):
        return 1e6
    return b


def construct():
    t0 = time.time()
    deadline = t0 + 165.0
    rng = np.random.default_rng(12345)

    # seed direction with c0 normalized to 1
    seed = np.array([(1.0 / 25.0) / (-113.0 / 100.0),
                     (1.0 / 3240.0) / (-113.0 / 100.0)])
    starts = [seed]
    for _ in range(120):
        starts.append(rng.uniform([-10.0, -1.0], [10.0, 1.0]))

    best_val = np.inf
    best_v = None
    for s in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(_obj, np.array(s), method='Nelder-Mead',
                           options={'maxiter': 250, 'xatol': 1e-12,
                                    'fatol': 1e-12, 'adaptive': True})
        except Exception:
            continue
        if np.isfinite(res.fun) and res.fun < best_val:
            best_val = res.fun
            best_v = res.x

    # high-precision polish of the best basin
    if best_v is not None and time.time() < deadline - 5.0:
        try:
            res = minimize(_obj, best_v, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-14,
                                    'fatol': 1e-14, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_v = res.x
        except Exception:
            pass

    if best_v is None:
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    return np.array([1.0, float(best_v[0]), float(best_v[1])])
