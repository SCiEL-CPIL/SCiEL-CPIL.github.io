"""Optimization-based program: minimize r^2/(2*pi) by optimizing coefficients."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import factorial as sp_factorial


def _h2n_even(n):
    """H_{2n}(0) = (-1)^n (2n)! / n!  (physicists' Hermite)."""
    return ((-1) ** n) * sp_factorial(2 * n, exact=True) / sp_factorial(n, exact=True)


# H0(0)=1, H4(0)=-12, H8(0)=1008, H12(0)=-103320
_H0_0 = 1
_H4_0 = -12
_H8_0 = 1008
_H12_0 = -103320


def _coeffs_to_q(c0, c1, c2):
    """Return (c3, q) where g=c0 H0+c1 H4+c2 H8+c3 H12, g(0)=0, q=g/x^2."""
    c3 = -(_H0_0 * c0 + _H4_0 * c1 + _H8_0 * c2) / _H12_0
    # g(x) = c0 + c1(16x^4-48x^2+12) + c2(256x^8-2304x^6+4320x^4-1512x^2+1008)
    #       + c3(4096x^12-129024x^10+1198080x^8-4644960x^6+6652800x^4-2126112x^2+103320)
    a12 = 4096.0 * c3
    a10 = -129024.0 * c3
    a8 = 256.0 * c2 + 1198080.0 * c3
    a6 = -2304.0 * c2 - 4644960.0 * c3
    a4 = 16.0 * c1 + 4320.0 * c2 + 6652800.0 * c3
    a2 = -48.0 * c1 - 1512.0 * c2 - 2126112.0 * c3
    return c3, np.array([a12, a10, a8, a6, a4, a2])


def _find_r(q):
    """Largest real root of q at which q changes sign."""
    # Evaluate q on a grid to bracket sign changes
    x = np.linspace(1e-9, 10.0, 20001)
    y = np.polyval(q, x)
    sign = np.sign(y)
    sign[sign == 0] = 1
    change = np.where(np.diff(sign) != 0)[0]
    best = -1.0
    for i in change:
        lo, hi = x[i], x[i + 1]
        if y[i] == 0:
            root = lo
        elif y[i + 1] == 0:
            root = hi
        else:
            try:
                root = np.roots(q)
                real_roots = [r.real for r in root if abs(r.imag) < 1e-9]
                # find root in (lo,hi)
                cands = [rr for rr in real_roots if lo - 1e-12 <= rr <= hi + 1e-12]
                if cands:
                    root = min(cands, key=lambda v: abs(v - (lo + hi) / 2))
                else:
                    # bisection
                    a, b = lo, hi
                    fa, fb = y[i], y[i + 1]
                    if fa * fb > 0:
                        continue
                    for _ in range(80):
                        m = (a + b) / 2
                        fm = np.polyval(q, m)
                        if fa * fm <= 0:
                            b, fb = m, fm
                        else:
                            a, fa = m, fm
                    root = (a + b) / 2
            except Exception:
                continue
        if root > best:
            best = root
    return best


def _objective(c):
    """Compute value = r^2/(2*pi), or a large penalty."""
    c0, c1, c2 = c
    c3, q = _coeffs_to_q(c0, c1, c2)
    # check g tends to negative limit -> replace by -g
    if q[0] < 0:
        q = -q
    r = _find_r(q)
    if r < 0:
        return 1e10
    return (r * r) / (2.0 * np.pi)


def _refine(c_best):
    """Local polish with Nelder-Mead."""
    try:
        res = minimize(_objective, c_best, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if res.fun < _objective(c_best):
            return res.x
    except Exception:
        pass
    return c_best


def construct():
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    seed_val = _objective(seed)

    best_c = seed.copy()
    best_val = seed_val

    # Differential evolution over a moderate box
    bounds = [(-2.0, 2.0), (-0.5, 0.5), (-0.005, 0.005)]
    try:
        res = differential_evolution(_objective, bounds, seed=42, maxiter=200,
                                     tol=1e-8, popsize=15, polish=True,
                                     mutation=(0.3, 1.0), recombination=0.8,
                                     init='sobol')
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x
    except Exception:
        pass

    # A few local refinements from the best and a couple of random restarts
    for start in [best_c, seed]:
        cand = _refine(np.array(start, dtype=float))
        v = _objective(cand)
        if v < best_val:
            best_val = v
            best_c = cand

    return np.array(best_c, dtype=float)
# EVOLVE-BLOCK-END
