"""Uncertainty inequality: tune Hermite-Gaussian to minimize the C4 bound.

g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2),  a12 = -a0  (so g(0)=0).
Under the unitary Fourier convention, H_n e^{-x^2/2} for n in {0,4,8,12} are
eigenfunctions with eigenvalue i^n = 1, so g is self-Fourier and the bound
A(g) A(hat g) = ||g||_1^2 (with ||g||_2 = 1 normalization).
We minimize ||g||_1 over the 3 free coefficients.
"""
import time
import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import differential_evolution, minimize

# Deadline guard (evaluator kills at 210s wall-clock; keep safety margin).
_T0 = time.time()
_DEADLINE = 210.0 - 25.0  # 25s safety margin


def _time_left():
    return _DEADLINE - (time.time() - _T0)


# Precompute the polynomial basis on a dense grid.
_X = np.linspace(-14.0, 14.0, 40001)
_GAUSS = np.exp(-0.5 * _X * _X)
_H0 = np.ones_like(_X)
_H4 = eval_hermite(4, _X)
_H8 = eval_hermite(8, _X)
_H12 = eval_hermite(12, _X)
_DX = _X[1] - _X[0]


def _poly(a0, a4, a8):
    # a12 = -a0 to enforce g(0)=0 (H0(0)=1, H12(0)=40320... wait, need g(0)=0).
    # g(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0)
    # H0(0)=1, H4(0)=12, H8(0)=168, H12(0)=40320.
    # The contract says "the degree-12 weight is fixed so g(0)=0".
    # We fix a12 so that g(0)=0 given the other three:
    # a12 = -(a0*1 + a4*12 + a8*168) / 40320.
    a12 = -(a0 * 1.0 + a4 * 12.0 + a8 * 168.0) / 40320.0
    return (a0 * _H0 + a4 * _H4 + a8 * _H8 + a12 * _H12) * _GAUSS


def _l1_norm(a0, a4, a8):
    g = _poly(a0, a4, a8)
    return np.trapz(np.abs(g), _X)


def _l2_norm_sq(a0, a4, a8):
    g = _poly(a0, a4, a8)
    return np.trapz(g * g, _X)


def _objective(v):
    a0, a4, a8 = v
    l2sq = _l2_norm_sq(a0, a4, a8)
    if l2sq <= 0.0:
        return 1e300
    # Normalize to ||g||_2 = 1, then ||g||_1 is the (square root of the) bound.
    scale = np.sqrt(l2sq)
    return _l1_norm(a0 / scale, a4 / scale, a8 / scale)


def construct():
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _objective(seed)
    best_x = seed.copy()

    def _update(v, val):
        nonlocal best_val, best_x
        if val < best_val:
            best_val = val
            best_x = np.asarray(v, dtype=float).copy()

    # Global search with differential evolution (robust to local minima).
    if _time_left() > 40.0:
        try:
            bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
            de = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                mutation=(0.4, 1.0),
                recombination=0.8,
                polish=False,
                init="latinhypercube",
                callback=None,
            )
            if de.fun < best_val:
                best_val = de.fun
                best_x = de.x.copy()
        except Exception:
            pass

    # Local polish from the best global point and from the seed.
    for start in (best_x.copy(), seed.copy()):
        if _time_left() < 15.0:
            break
        try:
            res = minimize(
                _objective,
                start,
                method="L-BFGS-B",
                options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-14},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Final: return the three coefficients of the best normalized direction.
    # Return them scaled back to a representative magnitude (the raw coefficients).
    # We return the direction (the actual a0,a4,a8 that achieved best_val).
    return np.asarray(best_x, dtype=float)
