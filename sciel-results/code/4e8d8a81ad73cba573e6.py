"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = 160.0  # safety margin below 210s kill

# ---------- Build the polynomial q(z) structure ----------
# g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12,  c3 = -(c0*H0(0)+c1*H4(0)+c2*H8(0))/H12(0)
# All H_{2k}(0) for k>=1 are 0, so g(0)=c0*H0(0)+c3*H12(0)=0  =>  g has factor x^2.
# q(x)=g(x)/x^2 is even in x; set z=x^2 => q is a degree-5 polynomial in z.
# We minimize the smallest positive real sign-changing root of q in x,
# equivalently the smallest positive real simple root in z.

try:
    import sympy as sp
    _x = sp.Symbol('x')
    _H0 = sp.hermite_poly(0, _x)
    _H4 = sp.hermite_poly(4, _x)
    _H8 = sp.hermite_poly(8, _x)
    _H12 = sp.hermite_poly(12, _x)
    # coefficients of each H as list [c0, c1, ..., ck] for x^0..x^n
    def _low_to_high(p):
        return np.array([float(c) for c in sp.Poly(p, _x).coeffs()])  # low->high
    _c0v = _low_to_high(_H0)
    _c1v = _low_to_high(_H4)
    _c2v = _low_to_high(_H8)
    _c3v = _low_to_high(_H12)
    # H12(0)
    _H12_0 = float(_c3v[0])
    _H0_0 = float(_c0v[0])
    _H4_0 = float(_c1v[0])  # 0
    _H8_0 = float(_c2v[0])  # 0
except Exception:
    # Fallback: hardcode physicists' Hermite polynomials (low->high coeffs)
    _H0_0 = 1.0
    _H4_0 = 0.0
    _H8_0 = 0.0
    _H12_0 = -1032160.0  # H12(0)
    # H0 = 1
    _c0v = np.array([1.0])
    # H4 = 16x^4 - 48x^2 + 12
    _c1v = np.array([12.0, 0.0, -48.0, 0.0, 16.0])
    # H8 = 256x^8 - 4480x^6 + 26880x^4 - 67200x^2 + 40320
    _c2v = np.array([40320.0, 0.0, -67200.0, 0.0, 26880.0, 0.0, -4480.0, 0.0, 256.0])
    # H12 = 4096x^12 - 147456x^10 + 2903040x^8 - 29030400x^6 + 145152000x^4 - 393216000x^2 + 1032160
    _c3v = np.array([1032160.0, 0.0, -393216000.0, 0.0, 145152000.0, 0.0, -29030400.0,
                     0.0, 2903040.0, 0.0, -147456.0, 0.0, 4096.0])
    _H12_0 = float(_c3v[0])

_deg = 12

def _g_coeffs(c0, c1, c2):
    # c3 from g(0)=0
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    g = np.zeros(_deg + 1)
    g += c0 * np.pad(_c0v, (0, _deg + 1 - len(_c0v)))
    g += c1 * np.pad(_c1v, (0, _deg + 1 - len(_c1v)))
    g += c2 * np.pad(_c2v, (0, _deg + 1 - len(_c2v)))
    g += c3 * np.pad(_c3v, (0, _deg + 1 - len(_c3v)))
    return g

def _q_coeffs(c0, c1, c2):
    g = _g_coeffs(c0, c1, c2)
    # drop the two lowest (x^0, x^1) terms
    return g[2:]

def _q_in_z(c0, c1, c2):
    q = _q_coeffs(c0, c1, c2)
    # even powers: index 0 -> z^0, index 2 -> z^1, ...
    z = q[0::2]
    return z

def _smallest_positive_simple_root(z_coeffs):
    """Return smallest positive real simple (sign-changing) root of poly z_coeffs (low->high).
    If none, return +inf."""
    # Remove leading zeros
    zc = z_coeffs.copy()
    while len(zc) > 1 and abs(zc[-1]) < 1e-30:
        zc = zc[:-1]
    if len(zc) < 2:
        return np.inf
    roots = np.roots(zc)
    best = np.inf
    for rt in roots:
        if abs(rt.imag) > 1e-9 * max(1.0, abs(rt.real)):
            continue
        zr = float(rt.real)
        if zr <= 1e-12:
            continue
        # check it's a simple root (sign change): evaluate derivative at zr
        d = np.polyder(np.flip(zc))  # high->low
        dval = np.polyval(d, zr)
        if abs(dval) < 1e-9:
            continue
        # also confirm q(zr) ~ 0
        qval = np.polyval(np.flip(zc), zr)
        if abs(qval) > 1e-6 * max(1.0, abs(zr)):
            continue
        if zr < best:
            best = zr
    return best

def _value(c0, c1, c2):
    zc = _q_in_z(c0, c1, c2)
    r = _smallest_positive_simple_root(zc)
    if not np.isfinite(r) or r <= 0:
        return np.inf
    return r / (2.0 * np.pi)

# ---------- Objective wrapper (finite, bounded) ----------
_BIG = 1e12
def _obj(params):
    try:
        v = _value(float(params[0]), float(params[1]), float(params[2]))
    except Exception:
        return _BIG
    if not np.isfinite(v):
        return _BIG
    return min(v, _BIG)

# ---------- Search ----------
def _seed():
    return np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])

def _local_refine(x0):
    """Nelder-Mead with tight tolerances, bounded-ish via penalty on big values."""
    try:
        res = minimize(_obj, x0, method='Nelder-Mead',
                       options={'maxiter': 400, 'xatol': 1e-12, 'fatol': 1e-14,
                                'adaptive': True})
        if res.success or (res.fun < _BIG):
            return res.x, res.fun
        return x0, _obj(x0)
    except Exception:
        return x0, _obj(x0)

def _polish(x0):
    """High-precision polish: use the active (smallest) root branch.
    Near the optimum the smallest root is an isolated simple root; we can
    do a tight Nelder-Mead with very small tolerances."""
    try:
        res = minimize(_obj, x0, method='Nelder-Mead',
                       options={'maxiter': 800, 'xatol': 1e-14, 'fatol': 1e-16,
                                'adaptive': True, 'initial_simplex': None})
        if np.isfinite(res.fun) and res.fun < _BIG:
            return res.x, res.fun
        return x0, _obj(x0)
    except Exception:
        return x0, _obj(x0)

def construct():
    global _T0
    try:
        rng = np.random.default_rng(12345)
        best_x = _seed().copy()
        best_v = _obj(best_x)
        if not np.isfinite(best_v):
            best_v = _BIG

        # 1) Global random + coarse grid multi-start
        starts = []
        # seed
        starts.append(best_x.copy())
        # random points in a reasonable box
        for _ in range(2000):
            if time.time() - _T0 > _DEADLINE - 40:
                break
            # scale-invariant; explore a box
            p = rng.uniform(-5, 5, size=3)
            starts.append(p)
        # also try points pushing one coefficient large to create small roots
        for a in [0.1, 0.5, 1, 2, 5]:
            for _ in range(200):
                if time.time() - _T0 > _DEADLINE - 40:
                    break
                p = rng.uniform(-a, a, size=3)
                starts.append(p)

        # Evaluate all starts (cheap-ish) and keep top candidates
        scored = []
        for p in starts:
            if time.time() - _T0 > _DEADLINE - 40:
                break
            v = _obj(p)
            if np.isfinite(v) and v < _BIG:
                scored.append((v, p.copy()))

        scored.sort(key=lambda t: t[0])
        top = scored[:20] if len(scored) >= 20 else scored

        # 2) Local refine each top candidate
        for v, p in top:
            if time.time() - _T0 > _DEADLINE - 15:
                break
            x, val = _local_refine(p)
            if val < best_v:
                best_v = val
                best_x = x.copy()

        # 3) Polish the best
        if time.time() - _T0 < _DEADLINE - 5:
            x, val = _polish(best_x)
            if val < best_v:
                best_v = val
                best_x = x.copy()

        return best_x
    except Exception:
        return _seed()
# EVOLVE-BLOCK-END
