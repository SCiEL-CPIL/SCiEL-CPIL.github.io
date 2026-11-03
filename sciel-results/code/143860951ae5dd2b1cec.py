"""Optimized search for Hermite-Gaussian coefficients minimizing the C4 upper bound."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

_T0 = time.time()
_DEADLINE = 180.0  # safety margin before 210s kill
_SQRT_PI = np.sqrt(np.pi)

# Precompute factorials up to 12
_FACT = [1]
for _i in range(1, 13):
    _FACT.append(_FACT[-1] * _i)


def _fact(n):
    return _FACT[n]


def _poly_mul(A, B):
    n, m = len(A), len(B)
    out = [0.0] * (n + m - 1)
    for i in range(n):
        ai = A[i]
        if ai == 0.0:
            continue
        for j in range(m):
            out[i + j] += ai * B[j]
    return out


def _poly_add(A, B):
    n = max(len(A), len(B))
    out = [0.0] * n
    for i in range(n):
        a = A[i] if i < len(A) else 0.0
        b = B[i] if i < len(B) else 0.0
        out[i] = a + b
    return out


def _poly_scale(A, s):
    return [a * s for a in A]


# Hermite polynomials (physicists) as ascending-power coefficient lists
_H0 = [1.0]
_H4 = [3.0, 0.0, -6.0, 0.0, 1.0]
_H8 = [105.0, 0.0, -420.0, 0.0, 210.0, 0.0, -28.0, 0.0, 1.0]
_H12 = [945.0, 0.0, -20160.0, 0.0, 30240.0, 0.0, -12600.0, 0.0, 1500.0, 0.0, -60.0, 0.0, 1.0]


def _poly_norm_sq(p):
    """Integral of p(x)^2 * exp(-x^2) dx = sqrt(pi) * sum_i p_i^2 * i!."""
    s = 0.0
    for i, a in enumerate(p):
        if a != 0.0:
            s += a * a * _FACT[i]
    return s * _SQRT_PI


def _bound(c0, c4, c8):
    """Compute the C4 upper bound A(f)*A(F) for the given coefficients."""
    c12 = -(c0 + 3.0 * c4 + 105.0 * c8) / 945.0

    p = _poly_add(_poly_add(_poly_add(_poly_scale(_H0, c0), _poly_scale(_H4, c4)),
                            _poly_scale(_H8, c8)),
                  _poly_scale(_H12, c12))

    n0 = _poly_norm_sq(p)
    if n0 <= 1e-30:
        return np.inf

    x2 = [0.0, 0.0, 1.0]
    x4 = _poly_mul(x2, x2)

    n2 = _poly_norm_sq(_poly_mul(p, x2))
    n4 = _poly_norm_sq(_poly_mul(p, x4))

    if n2 <= 0.0 or n4 <= 0.0:
        return np.inf

    A = np.sqrt(n0 / n2)
    B = np.sqrt(n2 / n4)
    return A * B


def construct():
    """Return three coefficients [c0, c4, c8] minimizing the C4 upper bound."""
    global _T0, _DEADLINE

    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best = seed.copy()
    best_val = _bound(*seed)

    def obj2(v):
        a, b = v
        return _bound(1.0, a, b)

    bounds = [(-50.0, 50.0), (-50.0, 50.0)]

    remaining = _DEADLINE - (time.time() - _T0)
    if remaining < 5.0:
        return best

    # Global search: differential evolution on the 2D ratio space
    try:
        result = differential_evolution(
            obj2, bounds,
            maxiter=300, popsize=20,
            tol=1e-10, mutation=(0.3, 1.2), recombination=0.8,
            seed=42, polish=True,
            init='sobol'
        )
        if result.fun < best_val:
            best_val = result.fun
            best = np.array([1.0, result.x[0], result.x[1]])
    except Exception:
        pass

    # Local refinement with Nelder-Mead from the best point
    remaining = _DEADLINE - (time.time() - _T0)
    if remaining > 10.0:
        try:
            res = minimize(
                lambda v: _bound(v[0], v[1], v[2]),
                best, method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 2000, 'maxfev': 5000}
            )
            if res.fun < best_val:
                best_val = res.fun
                best = res.x.copy()
        except Exception:
            pass

    # Multi-start local refinement to escape local minima
    remaining = _DEADLINE - (time.time() - _T0)
    if remaining > 10.0:
        starts = [
            [1.0, -0.04, 0.0003],
            [1.0, -0.02, 0.0001],
            [1.0, 0.0, 0.0],
            [1.0, -0.1, 0.001],
            [1.0, 0.05, -0.0005],
            [1.0, -0.5, 0.01],
            [1.0, 0.5, -0.01],
            [-1.0, 0.04, -0.0003],
        ]
        for st in starts:
            if time.time() - _T0 > _DEADLINE - 5.0:
                break
            try:
                res = minimize(
                    lambda v: _bound(v[0], v[1], v[2]),
                    st, method='Nelder-Mead',
                    options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 1000, 'maxfev': 3000}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best = res.x.copy()
            except Exception:
                pass

    if time.time() - _T0 > _DEADLINE - 2.0:
        return np.clip(best, -1e6, 1e6)

    return best
