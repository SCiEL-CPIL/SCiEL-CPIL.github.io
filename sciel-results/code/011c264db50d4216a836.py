"""Tune Hermite-Gaussian test function to reduce upper bound on C4."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Precompute Hermite polynomials H_0..H_12 (physicists', e^{-x^2/2} convention)
def _hermite_coeffs():
    H = [np.array([1.0])]
    for n in range(1, 13):
        H.append(2 * np.polyadd(
            np.polymul(np.array([2.0]), H[n - 1]),
            np.polyder(H[n - 1])))
    return H

_H = _hermite_coeffs()
_H0 = _H[0]
_H4 = _H[4]
_H8 = _H[8]
_H12 = _H[12]

# Coefficients of H^2 (even only)
_H0sq = np.polysqr(_H0)
_H4sq = np.polysqr(_H4)
_H8sq = np.polysqr(_H8)
_H12sq = np.polysqr(_H12)

# Cross products H_i * H_j (even only)
_H0_H4 = np.polymul(_H0, _H4)
_H0_H8 = np.polymul(_H0, _H8)
_H0_H12 = np.polymul(_H0, _H12)
_H4_H8 = np.polymul(_H4, _H8)
_H4_H12 = np.polymul(_H4, _H12)
_H8_H12 = np.polymul(_H8, _H12)


def _even_moments(poly):
    """m_{2k} = integral x^{2k} poly(x) dx for poly = sum c_n x^n (even n)."""
    c = np.asarray(poly, dtype=float)
    deg = len(c) - 1
    n = min(deg // 2, 6)
    out = np.zeros(n + 1)
    for k in range(n + 1):
        target = 2 * k
        if target < deg and target % 2 == 0:
            out[k] = c[deg - target]
    return out


def _gauss_odd(n):
    """G_{2n+1} = integral x^{2n+1} e^{-x^2/4} dx = 2^{2n} n! (for n>=0)."""
    return float(2.0 ** (2 * n) * np.math.factorial(n))


# Precompute even moments of each quadratic/cross polynomial
_M00 = _even_moments(_H0sq)
_M44 = _even_moments(_H4sq)
_M88 = _even_moments(_H8sq)
_M1212 = _even_moments(_H12sq)
_M04 = _even_moments(_H0_H4)
_M08 = _even_moments(_H0_H8)
_M012 = _even_moments(_H0_H12)
_M48 = _even_moments(_H4_H8)
_M412 = _even_moments(_H4_H12)
_M812 = _even_moments(_H8_H12)

_G = [_gauss_odd(k) for k in range(7)]  # G_1, G_3, ..., G_13
_SQRT2PI = float(np.sqrt(2.0 * np.pi))


def _bound_from_moments(m):
    """m[k] = integral x^{2k} H^2(x) dx. Return (A(g), A(hat g))."""
    I1 = np.sqrt(np.pi) * (m[0] + 2.0 * m[1] + 4.0 * m[2] + 8.0 * m[3]
                           + 16.0 * m[4] + 32.0 * m[5] + 64.0 * m[6])
    I2 = _SQRT2PI * (m[0] + 2.0 * _G[0] * m[1] + 2.0 * _G[1] * m[2]
                     + 2.0 * _G[2] * m[3] + 2.0 * _G[3] * m[4]
                     + 2.0 * _G[4] * m[5] + 2.0 * _G[5] * m[6])
    return I1, I2


def _objective(x):
    a0, a4, a8 = x
    a12 = -(a0 + a4 + a8) / 12.0
    m0 = (a0 * a0) * _M00 + (a4 * a4) * _M44 + (a8 * a8) * _M88 \
        + (a12 * a12) * _M1212 \
        + 2.0 * (a0 * a4) * _M04 + 2.0 * (a0 * a8) * _M08 \
        + 2.0 * (a0 * a12) * _M012 + 2.0 * (a4 * a8) * _M48 \
        + 2.0 * (a4 * a12) * _M412 + 2.0 * (a8 * a12) * _M812
    if m0[0] <= 1e-12:
        return 1e12
    I1, I2 = _bound_from_moments(m0)
    if I1 <= 1e-12 or I2 <= 1e-12:
        return 1e12
    return (I1 / np.sqrt(I2)) * (I2 / np.sqrt(I1))


_REF = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_REF_VAL = _objective(_REF)


def _polish(x0, deadline, iters=4000):
    from scipy.optimize import minimize
    best = x0.copy()
    best_v = _objective(x0)
    try:
        res = minimize(_objective, x0, method='L-BFGS-B',
                       options={'maxiter': iters, 'maxfun': 6000, 'ftol': 1e-14})
        if res.success or res.fun < best_v:
            if np.all(np.isfinite(res.x)):
                v = _objective(res.x)
                if v < best_v:
                    best = res.x
                    best_v = v
    except Exception:
        pass
    return best, best_v


def construct():
    deadline = time.time() + 150.0
    best = _REF.copy()
    best_v = _REF_VAL

    # 1) Global search: Differential Evolution in spherical coordinates
    try:
        from scipy.optimize import differential_evolution
        # Parameterize (a0,a4,a8) = r * (sin(th)*cos(phi), sin(th)*sin(phi), cos(th))
        def sphere_obj(p):
            if time.time() > deadline:
                return 1e12
            r, th, phi = p
            x = r * np.array([np.sin(th) * np.cos(phi),
                              np.sin(th) * np.sin(phi),
                              np.cos(th)])
            return _objective(x)

        def sphere_unwrap(p):
            r, th, phi = p
            return r * np.array([np.sin(th) * np.cos(phi),
                                 np.sin(th) * np.sin(phi),
                                 np.cos(th)])

        bounds = [(0.0, 2.0), (0.0, np.pi), (0.0, 2.0 * np.pi)]
        de = differential_evolution(sphere_obj, bounds,
                                    seed=12345, maxiter=60, popsize=12,
                                    tol=1e-6, mutation=(0.3, 1.2),
                                    recombination=0.8, init='sobol',
                                    polish=False, updating='deferred')
        if de.success and np.all(np.isfinite(de.x)):
            cand = sphere_unwrap(de.x)
            v = _objective(cand)
            if v < best_v:
                best = cand
                best_v = v
    except Exception:
        pass

    # 2) Local polish from best and from reference
    for start in (best, _REF):
        if time.time() > deadline:
            break
        c, v = _polish(start, deadline)
        if v < best_v:
            best = c
            best_v = v

    # 3) Tiny extra polish from the current best
    if time.time() < deadline - 5.0:
        c, v = _polish(best, deadline)
        if v < best_v:
            best = c
            best_v = v

    return np.array([best[0], best[1], best[2]])
# EVOLVE-BLOCK-END
