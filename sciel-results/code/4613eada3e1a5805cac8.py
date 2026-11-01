"""Rewritten program: analytic Fourier moments + 2-D algebraic optimization.

The Hermite-Gaussian basis is a Fourier eigenfunction basis, so the
frequency-space moments are closed-form algebraic functions of the
coefficients (no numerical integration).  The constraint g(0)=0 is linear in
the coefficients, so the degree-12 weight is eliminated analytically and the
upper bound A(f)*A(F) = ||g||_2^2 * ||hat g||_2^2 = ||g||_2^4 becomes a
pure function of the two free ratios (a4/a0, a8/a0).  We minimize that
function with a fast multi-start Nelder-Mead search guarded by a deadline.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

# ---- Fixed analytic constants -------------------------------------------------
# g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12)(x) * exp(-x^2/2),  physicists' Hn.
# ||Hn*exp(-x^2/2)||_2^2 = 2^n * n! * sqrt(pi)  (independent of the coefficient).
_SQRTPI = math.sqrt(math.pi)
_NORM = np.array([1.0, 24.0, 1024.0, 16384.0]) * _SQRTPI   # n = 0,4,8,12
# Hn(0): 1, 12, 168, 40320.  g(0)=0  =>  a0 + 12 a4 + 168 a8 + 40320 a12 = 0.
# => a12 = -(a0 + 12 a4 + 168 a8) / 40320.
# Fourier: F(Hn*exp(-x^2/2)) = e^{-i pi/4} * i^n * (Hn*exp(-x^2/2))  =>  |.|^2 = 2^n n! pi.
_FOUR = np.array([1.0, 24.0, 1024.0, 16384.0]) * math.pi
_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
# Seed bound ||g||_2^4 (the reference value; lower is better).
_SEED_BOUND = (float(np.dot(_SEED, _SEED * _NORM)) ** 2)


def _bound_from_coeffs(a0, a4, a8):
    """Upper bound A(f)*A(F) for given a0,a4,a8 (a12 eliminated analytically)."""
    a12 = -(a0 + 12.0 * a4 + 168.0 * a8) / 40320.0
    g = np.array([a0, a4, a8, a12])
    n2 = float(np.dot(g, g * _NORM))          # ||g||_2^2
    f2 = float(np.dot(g, g * _FOUR))          # ||hat g||_2^2  (== n2 by Parseval)
    return n2 * f2


def _objective(x):
    """Bound as a function of the two free ratios (p=a4/a0, q=a8/a0), a0=1."""
    return _bound_from_coeffs(1.0, x[0], x[1])


def _nm(x0, ftol=1e-12, maxiter=250, deadline=None):
    """Small self-contained Nelder-Mead (no scipy dependency)."""
    n = len(x0)
    best = x0.copy()
    bestf = _objective(best)
    # Build initial simplex.
    simplex = [best.copy()]
    for i in range(n):
        pt = best.copy()
        pt[i] = pt[i] + 0.2 if pt[i] == 0 else pt[i] * 1.2
        simplex.append(pt)
    vals = [_objective(p) for p in simplex]
    it = 0
    while it < maxiter:
        if deadline is not None and time.time() > deadline:
            break
        it += 1
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < ftol * (abs(vals[0]) + 1e-30):
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + 1.0 * (centroid - simplex[-1])
        fr = _objective(xr)
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = _objective(xe)
            if fe < fr:
                simplex[-1], vals[-1] = xe, fe
            else:
                simplex[-1], vals[-1] = xr, fr
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = _objective(xc)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = xc, fc
            else:
                for k in range(1, n + 1):
                    simplex[k] = best + 0.5 * (simplex[k] - best)
                    vals[k] = _objective(simplex[k])
        # track best
        m = int(np.argmin(vals))
        if vals[m] < bestf:
            bestf = vals[m]
            best = simplex[m].copy()
    return best, bestf


def construct():
    # Hard deadline guard: return best found with a safety margin before 180 s.
    deadline = time.time() + 170.0

    best_x = _SEED.copy()
    best_val = _SEED_BOUND

    # Coarse multi-start grid (fast, analytic objective).
    grid = []
    for a4 in np.linspace(-0.5, 0.5, 13):
        for a8 in np.linspace(-0.12, 0.12, 11):
            grid.append((float(a4), float(a8)))
    for p, q in grid:
        if time.time() > deadline:
            break
        v = _objective(np.array([p, q]))
        if v < best_val:
            best_val = v
            best_x = np.array([p, q])

    # Refine from the best grid point and a handful of promising neighbours.
    starts = [best_x.copy(), _SEED.copy()]
    for p, q in grid:
        d = (p - best_x[0]) ** 2 + (q - best_x[1]) ** 2
        if d < 0.02:
            starts.append(np.array([p, q]))
    for s in starts:
        if time.time() > deadline:
            break
        x, f = _nm(s, deadline=deadline)
        if f < best_val:
            best_val = f
            best_x = x

    # Map the optimized ratios (a4/a0, a8/a0) back to absolute coefficients
    # with a0 = 1 (the bound is scale-invariant, so only the ratios matter).
    a0, a4, a8 = 1.0, float(best_x[0]), float(best_x[1])
    a12 = -(a0 + 12.0 * a4 + 168.0 * a8) / 40320.0
    return np.array([a0, a4, a8, a12])
# EVOLVE-BLOCK-END
