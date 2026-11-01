"""Uncertainty inequality: tune Hermite-Gaussian test function to minimize A(f)*A(F(f))."""
# EVOLVE-BLOCK-START
import time

import numpy as np
from scipy.special import eval_hermite
from scipy.integrate import quad
from scipy.optimize import differential_evolution

# ---------- Fourier convention ----------
# g(x) = P(x) * exp(-x^2/2)
# Fourier transform (unitary, angular frequency):
#   F(k) = (1/sqrt(2*pi)) * integral g(x) e^{-i k x} dx
# For even g, F(k) = (1/sqrt(2*pi)) * integral g(x) cos(k x) dx
# The Gaussian and Hermite basis are self-dual, so F(g) is also a Hermite-Gaussian
# combination with the same coefficients.

_T0 = time.time()
_DEADLINE = _T0 + 195.0  # safety margin well under the 210 s kill limit
_PI = np.pi
_SQRT2PI = np.sqrt(2.0 * _PI)


def _hermite(x, n):
    """Physicist's Hermite polynomial H_n(x) via recurrence (robust for small n)."""
    h0 = np.ones_like(x, dtype=float)
    h1 = 2.0 * x
    if n == 0:
        return h0
    if n == 1:
        return h1
    hn2 = h0
    hn1 = h1
    for k in range(2, n + 1):
        hn = 2.0 * x * hn1 - 2.0 * (k - 1) * hn2
        hn2, hn1 = hn1, hn
    return hn1


def _g_of_x(x, u, v):
    """Test function g(x) = (H0 + u*H4 + v*H8 + w*H12) * exp(-x^2/2), w chosen so g(0)=0."""
    h0 = _hermite(x, 0)
    h4 = _hermite(x, 4)
    h8 = _hermite(x, 8)
    h12 = _hermite(x, 12)
    # g(0) = (1 + u*H4(0) + v*H8(0) + w*H12(0)) * 1 = 0
    h4_0 = 2.0 ** 4 * 3 * 15 / 24.0  # H4(0)
    h8_0 = 2.0 ** 8 * 10395 / 40320.0  # H8(0)
    h12_0 = 2.0 ** 12 * 2027025 / 479001600.0  # H12(0)
    w = -(1.0 + u * h4_0 + v * h8_0) / h12_0
    poly = h0 + u * h4 + v * h8 + w * h12
    return poly * np.exp(-0.5 * x * x)


def _a_of_g(u, v, x_max=40.0, limit=500):
    """A(g) = integral |g(x)| dx (numerical)."""
    def integrand(x):
        return abs(float(_g_of_x(np.array([x]), u, v)[0]))
    val, _err = quad(integrand, 0.0, x_max, limit=limit, epsabs=1e-10, epsrel=1e-10)
    return 2.0 * val  # even function


def _a_of_f(u, v, k_max=40.0, limit=500):
    """A(F(g)) = integral |F(g)(k)| dk, where F(g) is the unitary Fourier transform."""
    def integrand(k):
        gk = _g_of_x(np.array([k]), u, v)[0]  # self-dual: F(g)(k) ~ g(k) up to constant
        return abs(float(gk)) / _SQRT2PI
    val, _err = quad(integrand, 0.0, k_max, limit=limit, epsabs=1e-10, epsrel=1e-10)
    return 2.0 * val  # even function


def _objective(uv):
    """Return A(g) * A(F(g)) for the Hermite combination with ratios u, v."""
    u, v = uv
    ag = _a_of_g(u, v)
    af = _a_of_f(u, v)
    return ag * af


def _score_from_uv(uv):
    """Score = seed_value / raw_value (1.0 == seed program)."""
    raw = _objective(uv)
    if not np.isfinite(raw) or raw <= 0:
        return 0.0
    # seed raw value from the prior state-of-the-art coefficients
    seed_raw = _objective_from_coeffs(-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0)
    if seed_raw <= 0 or not np.isfinite(seed_raw):
        return 0.0
    return seed_raw / raw


def _objective_from_coeffs(a0, a4, a8):
    """Compute A(g)*A(F(g)) from raw coefficients (a0, a4, a8); a12 fixed by g(0)=0."""
    if a0 == 0:
        return np.inf
    u = a4 / a0
    v = a8 / a0
    return _objective([u, v])


def construct():
    # Reference (seed) raw value, computed once for scoring.
    seed_raw = _objective_from_coeffs(-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0)

    best_uv = np.array([-113.0 / 100.0, 1.0 / 25.0])  # placeholder
    # Initialize best with the seed's ratios: a0=-1.13, a4=0.04, a8=1/3240
    best_uv = np.array([0.04 / -1.13, (1.0 / 3240.0) / -1.13])
    best_raw = _objective(best_uv)
    if not np.isfinite(best_raw) or best_raw <= 0:
        best_raw = seed_raw if np.isfinite(seed_raw) else 1.0

    def _maybe_update(uv, raw):
        nonlocal best_uv, best_raw
        if np.isfinite(raw) and raw > 0 and raw < best_raw:
            best_uv = np.array(uv)
            best_raw = raw

    # ---------- Global optimization via differential evolution ----------
    bounds = [(-50.0, 50.0), (-50.0, 50.0)]
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            maxiter=120,
            popsize=12,
            tol=1e-6,
            mutation=(0.3, 1.2),
            recombination=0.7,
            seed=42,
            polish=True,
            init='latinhypercube',
            updating='deferred',
            callback=lambda xk, convergence: (
                _maybe_update(xk, _objective(xk)),
                False if time.time() > _DEADLINE - 30.0 else True
            ),
        )
        if result is not None and result.x is not None:
            _maybe_update(result.x, _objective(result.x))
    except Exception:
        pass

    # ---------- Local refinement with Nelder-Mead from several starts ----------
    from scipy.optimize import minimize
    starts = [best_uv.copy()]
    # Add a handful of deterministic + random starts within a tighter box.
    rng = np.random.default_rng(0)
    for _ in range(8):
        starts.append(rng.uniform(-10.0, 10.0, size=2))
    # A few structured starts near the seed and around the origin.
    starts.append(np.array([0.0, 0.0]))
    starts.append(np.array([-0.1, 0.0]))
    starts.append(np.array([0.0, -0.1]))

    for s in starts:
        if time.time() > _DEADLINE - 20.0:
            break
        try:
            res = minimize(_objective, s, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.x is not None:
                _maybe_update(res.x, _objective(res.x))
        except Exception:
            pass

    # ---------- Final scoring ----------
    best_raw = _objective(best_uv)
    if not np.isfinite(best_raw) or best_raw <= 0:
        # Fall back to the seed coefficients directly.
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    # Convert optimal ratios back to coefficients. Choose a0 = 1 for the returned triple;
    # the product A(f)*A(F(f)) is scale-invariant, so any a0 gives the same score.
    u, v = best_uv
    return np.array([1.0, u, v])
# EVOLVE-BLOCK-END
