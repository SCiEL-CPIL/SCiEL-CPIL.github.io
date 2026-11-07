"""Optimize the two free coefficient ratios to minimize the largest positive
simple root of the reduced polynomial, hence minimizing value = r^2/(2*pi)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

# Coefficients of H_{12}, H_8, H_4, H_0 expressed as polynomials in y = x^2,
# ordered from y^6 down to y^0 (physicists' Hermite polynomials).
_H12 = np.array([4096.0, -46080.0, 205824.0, -387072.0, 322560.0, -137280.0, 137280.0])
_H8 = np.array([0.0, 0.0, 256.0, -1344.0, 2688.0, -1344.0, 112.0])
_H4 = np.array([0.0, 0.0, 0.0, 0.0, 16.0, -48.0, 12.0])
_H0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
_CONST = 137280.0


def _build_coeffs(p, q_):
    """Coeffs (y^6..y^0) of P(y) = g(x) with c0=1, c1=p, c2=q_, c3 fixed by g(0)=0."""
    c3 = -(1.0 + 12.0 * p + 112.0 * q_) / _CONST
    return _H12 * c3 + _H8 * q_ + _H4 * p + _H0


def _largest_positive_simple_root(c):
    """Largest positive root of P at which P changes sign (odd multiplicity).
    Returns a large penalty if none exists."""
    # trim negligible leading coefficients
    nz = np.where(np.abs(c) > 1e-14 * max(1.0, np.max(np.abs(c))))[0]
    if len(nz) == 0:
        return 1e18
    c = c[nz[0]:]
    try:
        rts = np.roots(c)
    except Exception:
        return 1e18
    best = -1.0
    for rt in rts:
        if abs(rt.imag) > 1e-7 * max(1.0, abs(rt.real)):
            continue
        r = rt.real
        if r <= 1e-10:
            continue
        d = 1e-6 * max(1.0, abs(r))
        try:
            a = np.polyval(c, r - d)
            b = np.polyval(c, r + d)
        except Exception:
            continue
        if a * b < 0.0 and r > best:
            best = r
    return best if best > 0.0 else 1e18


def _objective(x):
    p, q_ = x
    return _largest_positive_simple_root(_build_coeffs(p, q_))


def _seed_ratios():
    c0, c1, c2 = -113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0
    return np.array([c1 / c0, c2 / c0])


def _smooth_objective(x, beta=100.0):
    """A smooth surrogate that approximates the max-over-roots objective.
    Uses a log-sum-exp style soft-max over candidate positive real roots."""
    p, q_ = x
    c = _build_coeffs(p, q_)
    nz = np.where(np.abs(c) > 1e-14 * max(1.0, np.max(np.abs(c))))[0]
    if len(nz) == 0:
        return 1e18
    c = c[nz[0]:]
    try:
        rts = np.roots(c)
    except Exception:
        return 1e18
    
    # Collect positive real roots with sign change
    candidates = []
    for rt in rts:
        if abs(rt.imag) > 1e-7 * max(1.0, abs(rt.real)):
            continue
        r = rt.real
        if r <= 1e-10:
            continue
        d = 1e-6 * max(1.0, abs(r))
        try:
            a = np.polyval(c, r - d)
            b = np.polyval(c, r + d)
        except Exception:
            continue
        if a * b < 0.0:
            candidates.append(r)
    
    if not candidates:
        return 1e18
    
    candidates = np.array(candidates)
    # Soft-max: approximate max with log-sum-exp
    max_c = np.max(candidates)
    if max_c < 1e-10:
        return 1e18
    # Use a smooth approximation: for large beta, this approaches the max
    exp_vals = np.exp(-beta * (max_c - candidates))
    return max_c - np.log(np.sum(exp_vals)) / beta


def construct():
    t0 = time.time()
    deadline = 175.0  # safety margin well under the 210s kill

    seed = _seed_ratios()
    best_x = seed.copy()
    best_val = _objective(best_x)

    # Phase 1: Differential evolution with multiple boxes
    boxes = [
        [(-3.0, 3.0), (-0.1, 0.1)],
        [(-1.0, 1.0), (-0.02, 0.02)],
        [(-8.0, 8.0), (-0.3, 0.3)],
    ]

    for bi, bounds in enumerate(boxes):
        if time.time() > deadline - 25.0:
            break
        try:
            res = differential_evolution(
                _objective, bounds, maxiter=40, popsize=18,
                tol=1e-9, seed=bi, polish=False, init="sobol",
                updating="deferred",
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
        if time.time() > deadline - 20.0:
            break

    # Phase 2: Smooth surrogate-based local optimization
    if time.time() < deadline - 10.0:
        try:
            # Use the smooth objective for a more robust local search
            res_smooth = minimize(_smooth_objective, best_x, method="Nelder-Mead",
                                 options={"maxiter": 3000, "xatol": 1e-12, "fatol": 1e-12})
            if res_smooth.fun < best_val:
                # Verify with the true objective
                true_val = _objective(res_smooth.x)
                if true_val < best_val:
                    best_val = true_val
                    best_x = res_smooth.x
        except Exception:
            pass

    # Phase 3: Final high-precision polish with the true objective
    if time.time() < deadline - 5.0:
        try:
            res = minimize(_objective, best_x, method="Nelder-Mead",
                           options={"maxiter": 4000, "xatol": 1e-12, "fatol": 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # Final safety check against the seed
    if _objective(seed) < best_val:
        best_x = seed

    p, q_ = best_x
    # c0 is arbitrary (scale/sign invariance); use 1.0
    return [1.0, float(p), float(q_)]
# EVOLVE-BLOCK-END
