"""Optimize Hermite-Gaussian coefficients to reduce the C4 uncertainty bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Physicists' Hermite polynomials
def H(n, x):
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    if n == 2:
        return 4 * x**2 - 2
    if n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    if n == 8:
        return (256 * x**8 - 1344 * x**6 + 2688 * x**4 - 1344 * x**2 + 168)
    if n == 12:
        return (4096 * x**12 - 30720 * x**10 + 70400 * x**8 - 84480 * x**6
                + 49152 * x**4 - 10752 * x**2 + 4608)
    raise ValueError(n)

# Quadrature points and weights for ∫ f(x) e^{-2x^2} dx and ∫ f(x) e^{-x^2} dx
# Use Gauss-Hermite adapted: for e^{-a x^2}, scale. We'll just use dense quadrature.
_x = np.linspace(-8, 8, 40001)
_dx = _x[1] - _x[0]
_w2 = np.exp(-2 * _x**2) * _dx   # weight for e^{-2x^2}
_w1 = np.exp(-_x**2) * _dx       # weight for e^{-x^2}

# Precompute H values on grid
_H0 = H(0, _x)
_H4 = H(4, _x)
_H8 = H(8, _x)
_H12 = H(12, _x)


def _P(a0, a4, a8):
    a12 = -(a0 + 12.0 * a4 + 168.0 * a8) / 4608.0
    return a0 * _H0 + a4 * _H4 + a8 * _H8 + a12 * _H12


def _bound(a0, a4, a8):
    P = _P(a0, a4, a8)
    num = np.sum(P**4 * _w2)
    den = np.sum(P**2 * _w1)
    if den <= 0:
        return np.inf
    return num / (den**2)


def construct():
    t0 = time.time()
    deadline = 180.0  # safety margin before 210s kill

    # Seed / prior state of the art
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best = seed.copy()
    best_val = _bound(*best)

    # Multi-start coordinate descent + Nelder-Mead polish
    starts = [seed.copy()]
    # Add a grid of perturbations around the seed
    rng = np.random.default_rng(0)
    for _ in range(60):
        starts.append(seed + rng.normal(0, 0.05, 3))
    for _ in range(40):
        starts.append(seed + rng.normal(0, 0.2, 3))
    for _ in range(20):
        starts.append(seed + rng.normal(0, 1.0, 3))

    for s in starts:
        if time.time() - t0 > deadline:
            break
        x = s.copy()
        # Coordinate descent: for each coordinate, do a 1-D line search
        for it in range(60):
            if time.time() - t0 > deadline:
                break
            improved = False
            for k in range(3):
                # sample along coordinate k
                base = x.copy()
                vals = []
                pts = []
                # local bracket around current value
                step = max(1e-4, 0.05 * (abs(base[k]) + 1.0))
                for t in np.linspace(-4, 4, 41):
                    cand = base.copy()
                    cand[k] = base[k] + t * step
                    vals.append(_bound(*cand))
                    pts.append(cand[k])
                vals = np.array(vals)
                pts = np.array(pts)
                j = int(np.argmin(vals))
                if vals[j] < best_val - 1e-12:
                    x = base.copy()
                    x[k] = pts[j]
                    best_val = vals[j]
                    best = x.copy()
                    improved = True
                # refine around best point with finer grid
                lo = pts[max(0, j - 1)]
                hi = pts[min(len(pts) - 1, j + 1)]
                for t in np.linspace(lo, hi, 21):
                    cand = base.copy()
                    cand[k] = t
                    v = _bound(*cand)
                    if v < best_val - 1e-12:
                        best_val = v
                        best = cand.copy()
                        improved = True
            if not improved:
                break

    # Nelder-Mead polish on best
    if time.time() - t0 < deadline - 20:
        res = minimize(_bound, best, method='Nelder-Mead',
                       options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14,
                                'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best = res.x

    # Final high-precision polish: tight Nelder-Mead
    if time.time() - t0 < deadline - 5:
        res = minimize(_bound, best, method='Nelder-Mead',
                       options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-16,
                                'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best = res.x

    return np.array([best[0], best[1], best[2]])
# EVOLVE-BLOCK-END
