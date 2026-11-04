"""Signed equal-width step functions minimizing max |autoconv| / (integral)^2.
Uses a continuous Zadoff-Chu-like chirp parameterization optimized via Brent's method.
"""
import numpy as np
from scipy.optimize import minimize_scalar
import time

N = 400

# EVOLVE-BLOCK-START

def _chirp_sequence(beta, N=N):
    """Construct a real sequence from a complex Zadoff-Chu-like chirp.
    x[n] = cos(2*pi*beta*n^2 / N)  for n=0..N-1
    This has near-constant magnitude and flat spectrum, minimizing autoconvolution peaks.
    """
    n = np.arange(N, dtype=np.float64)
    phase = 2.0 * np.pi * beta * n * n / N
    return np.cos(phase)


def _score_sequence(x):
    """Compute max |autoconvolution| / (sum x)^2.
    Autoconvolution: (x * x)[k] = sum_n x[n] x[n+k]
    We compute full linear convolution and take the max absolute value.
    The integral is sum(x). We divide by (sum(x))^2.
    If sum(x) is near zero, the ratio blows up, so we need to be careful.
    """
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    # Full linear autoconvolution
    ac = np.convolve(x, x, mode='full')
    max_abs = np.max(np.abs(ac))
    return max_abs / (s * s)


def _objective(beta):
    """Objective for scalar optimization: score of the chirp sequence with parameter beta."""
    x = _chirp_sequence(beta)
    return _score_sequence(x)


def construct():
    """Construct the optimal signed step function.
    Searches over the chirp parameter beta to minimize the C3' score.
    """
    t0 = time.time()
    deadline = 110.0  # safety margin before 120s kill

    # Initial best
    best_beta = 0.0
    best_score = _objective(best_beta)
    best_x = _chirp_sequence(best_beta)

    # Try a grid of beta values first to get a good starting point
    beta_grid = np.linspace(0.0, 1.0, 50)
    for b in beta_grid:
        if time.time() - t0 > deadline:
            break
        sc = _objective(b)
        if sc < best_score:
            best_score = sc
            best_beta = b
            best_x = _chirp_sequence(b)

    # Now refine with Brent's method in a few intervals
    # The objective is periodic-ish in beta, so try multiple local minima
    intervals = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    for (lo, hi) in intervals:
        if time.time() - t0 > deadline:
            break
        try:
            res = minimize_scalar(_objective, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-8, 'maxiter': 200})
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_beta = res.x
                best_x = _chirp_sequence(best_beta)
        except Exception:
            pass

    # Also try negative beta (equivalent to flipping sign of quadratic phase)
    for (lo, hi) in [(-0.5, -0.25), (-0.25, 0.0)]:
        if time.time() - t0 > deadline:
            break
        try:
            res = minimize_scalar(_objective, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-8, 'maxiter': 200})
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_beta = res.x
                best_x = _chirp_sequence(best_beta)
        except Exception:
            pass

    # Additional refinement: try a finer local search around the best beta found
    if time.time() - t0 < deadline:
        delta = 0.05
        for _ in range(5):
            if time.time() - t0 > deadline:
                break
            lo = best_beta - delta
            hi = best_beta + delta
            if lo < -1.0 or hi > 1.0:
                break
            try:
                res = minimize_scalar(_objective, bounds=(lo, hi), method='bounded',
                                      options={'xatol': 1e-10, 'maxiter': 300})
                if res.success and res.fun < best_score:
                    best_score = res.fun
                    best_beta = res.x
                    best_x = _chirp_sequence(best_beta)
                    delta *= 0.5
                else:
                    break
            except Exception:
                break

    # Ensure we return a valid sequence
    if best_x is None or not np.all(np.isfinite(best_x)):
        return np.ones(N)

    return best_x

# EVOLVE-BLOCK-END
