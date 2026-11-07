"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Physicist's Hermite polynomials, descending power order
_H0 = np.array([1.0])
_H4 = np.array([16.0, 0.0, -48.0, 0.0, 12.0])
_H8 = np.array([256.0, 0.0, -4608.0, 0.0, 23040.0, 0.0, -34560.0, 0.0, 10080.0])
_H12 = np.array([4096.0, 0.0, -147456.0, 0.0, 1857984.0, 0.0, -10321920.0, 0.0, 23950080.0, 0.0, -21258240.0, 0.0, 4790016.0])


def _build_q(c0, c1, c2):
    L = len(_H12)  # 13
    def pad(a):
        o = np.zeros(L); o[:len(a)] = a; return o
    c3 = -(c0*_H0[-1] + c1*_H4[-1] + c2*_H8[-1]) / _H12[-1]
    g = c0*pad(_H0) + c1*pad(_H4) + c2*pad(_H8) + c3*_H12
    return g[:L-1]


def _largest_scc(q):
    rs = np.roots(q)
    rr = rs[np.abs(rs.imag) < 1e-10].real
    out = []
    for r in rr:
        e = 1e-8*(1+abs(r))
        if np.polyval(q, r-e)*np.polyval(q, r+e) < 0:
            out.append(r)
    return max(out) if out else None


def _val(x):
    r = _largest_scc(_build_q(*x))
    return r*r/(2*np.pi) if r is not None else 1e10


def _obj(x):
    try:
        v = _val(x)
        return v if np.isfinite(v) else 1e10
    except:
        return 1e10


def construct():
    t0 = time.time()
    dl = t0 + 170
    bx = np.array([-1.13, 0.04, 1/3240.0])
    bv = _val(bx)
    res = minimize(_obj, bx, method='Nelder-Mead', options={'maxiter':5000,'xatol':1e-10,'fatol':1e-12})
    if res.fun < bv: bv, bx = res.fun, res.x.copy()
    if time.time() > dl - 30: return bx
    np.random.seed(42)
    for _ in range(20):
        if time.time() > dl - 30: break
        x0 = bx + np.random.randn(3)*0.3
        r = minimize(_obj, x0, method='Nelder-Mead', options={'maxiter':2000,'xatol':1e-10,'fatol':1e-12})
        if r.fun < bv: bv, bx = r.fun, r.x.copy()
    r = minimize(_obj, bx, method='Nelder-Mead', options={'maxiter':10000,'xatol':1e-12,'fatol':1e-14})
    if r.fun < bv: bx = r.x.copy()
    return bx
# EVOLVE-BLOCK-END
