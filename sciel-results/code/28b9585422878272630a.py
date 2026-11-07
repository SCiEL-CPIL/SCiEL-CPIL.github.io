"""Optimize the three Hermite coefficients to minimize r^2/(2*pi).

g(x) = (x^2/240)*q(x) with q(x) = a x^8 + b x^6 + c x^4 + d x^2 + e,
an even polynomial. The relevant root is the largest-magnitude real root,
so value = r^2/(2*pi) and minimizing value == minimizing r^2.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
from mpmath import mp, polyroots, mpf

_START = time.time()
_DEADLINE = _START + 190.0  # safety margin under the 210 s wall clock
_SEED = (-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0)
_PI = np.pi

# ---------------------------------------------------------------------------
# Exact rational constants (as floats, then promoted to mpf for high precision)
# ---------------------------------------------------------------------------
def _hermite_value(n, x):
    # physicists' Hermite H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}
    # = sum_{k=0}^{n} (-1)^k * n! / k! * (2x)^{n-2k}
    total = 0.0
    for k in range(n + 1):
        coeff = (-1.0) ** k * (float(factorial(n, exact=True)) / float(factorial(k, exact=True)))
        total += coeff * (2.0 * x) ** (n - 2 * k)
    return total


_H0 = _hermite_value(0, 0.0)
_H4 = _hermite_value(4, 0.0)
_H8 = _hermite_value(8, 0.0)
_H12 = _hermite_value(12, 0.0)

# q coefficients as affine functions of (c0,c1,c2)
_A0 = _H8 / _H12
_A1 = -_H4 / _H12
_A2 = -_H0 / _H12
_B0 = -_H8 / 240.0
_B1 = _H4 / 240.0
_B2 = _H0 / 240.0
_C0 = _H8 / 240.0
_C1 = -_H4 / 240.0
_C2 = _H0 / 240.0
_D0 = -_H8 / 240.0
_D1 = _H4 / 240.0
_D2 = -_H0 / 240.0
_E0 = _H8 / 240.0
_E1 = -_H4 / 240.0
_E2 = _H0 / 240.0


def _coeffs(c0, c1, c2):
    a = _A0 * c0 + _A1 * c1 + _A2 * c2
    b = _B0 * c0 + _B1 * c1 + _B2 * c2
    c = _C0 * c0 + _C1 * c1 + _C2 * c2
    d = _D0 * c0 + _D1 * c1 + _D2 * c2
    e = _E0 * c0 + _E1 * c1 + _E2 * c2
    return a, b, c, d, e


def _objective(c):
    c0, c1, c2 = c
    a, b, cc, d, e = _coeffs(c0, c1, c2)
    if abs(a) < 1e-14:
        a = 1e-14 if a >= 0 else -1e-14
    poly = np.array([a, 0.0, b, 0.0, cc, 0.0, d, 0.0, e])
    roots = np.roots(poly)
    real = roots[np.abs(roots.imag) < 1e-8]
    if real.size == 0:
        return 1e12
    r2 = float(np.max(real * real))
    return r2


def _objective_mp(c, dps):
    c0, c1, c2 = (mpf(v) for v in c)
    a = _A0 * c0 + _A1 * c1 + _A2 * c2
    b = _B0 * c0 + _B1 * c1 + _B2 * c2
    cc = _C0 * c0 + _C1 * c1 + _C2 * c2
    d = _D0 * c0 + _D1 * c1 + _D2 * c2
    e = _E0 * c0 + _E1 * c1 + _E2 * c2
    if abs(a) < mpf("1e-30"):
        a = mpf("1e-30")
    poly = [a, mpf(0), b, mpf(0), cc, mpf(0), d, mpf(0), e]
    roots = polyroots(poly)
    best = mpf(0)
    for rt in roots:
        if abs(rt.imag) < mpf("1e-25"):
            r2 = (rt.real) ** 2
            if r2 > best:
                best = r2
    return float(best)


def _polish(c, time_budget):
    """High-precision local search around c."""
    old_dps = mp.dps
    best_c = list(c)
    best_val = _objective_mp(best_c, 50)
    try:
        mp.dps = 50
        base = [mpf(v) for v in best_c]
        for _ in range(6):
            if time.time() > _START + time_budget:
                break
            improved = False
            for i in range(3):
                for delta in (mpf("1e-6"), mpf("-1e-6"), mpf("1e-7"), mpf("-1e-7")):
                    trial = list(base)
                    trial[i] = base[i] + delta
                    val = _objective_mp(trial, 50)
                    if val < best_val - 1e-15:
                        best_val = val
                        base = trial
                        improved = True
            if not improved:
                break
    finally:
        mp.dps = old_dps
    best_c = [float(v) for v in base]
    return best_c, best_val


def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 190.0

    # 1) Multistart L-BFGS-B from the seed and a handful of random starts.
    best_c = list(_SEED)
    best_val = _objective(best_c)

    rng = np.random.default_rng(0)
    starts = [np.array(_SEED, dtype=float)]
    for scale in (0.1, 0.5, 1.0, 2.0):
        for _ in range(6):
            starts.append(np.array(_SEED, dtype=float) + scale * rng.normal(size=3))

    for s in starts:
        if time.time() > _START + 150.0:
            break
        try:
            res = minimize(_objective, s, method="L-BFGS-B",
                           options={"maxiter": 400, "maxls": 40, "ftol": 1e-14})
        except Exception:
            continue
        if res.fun < best_val:
            best_val = float(res.fun)
            best_c = [float(x) for x in res.x]

    # 2) High-precision polish around the best point found.
    if time.time() < _START + 185.0:
        try:
            pc, pv = _polish(best_c, 185.0)
            if pv < best_val:
                best_c, best_val = pc, pv
        except Exception:
            pass

    return np.array(best_c, dtype=float)
# EVOLVE-BLOCK-END
