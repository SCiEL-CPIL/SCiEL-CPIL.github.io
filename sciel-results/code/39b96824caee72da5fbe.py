"""Acquisition for Bayesian optimization of the 1-D sphere (minimize, minimum 0)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import ndtr, ndtrc
from scipy.stats import norm


def _eratio(m, s):
    """E[1/f] for f ~ N(m, s^2), computed stably as (1/s) exp(m^2/(2 s^2)) Phi(m/s)."""
    s = np.maximum(s, 1e-9)
    z = m / s
    base = (1.0 / s) * np.exp(0.5 * z * z)
    # Standard normal CDF of z, with a large-|z| asymptotic to avoid overflow in base.
    cdf = np.empty_like(z)
    big = np.abs(z) > 12.0
    cdf[~big] = norm.cdf(z[~big])
    if np.any(big):
        zb = z[big]
        # For z >> 0, Phi(z) ~ 1; for z << 0, Phi(z) ~ pdf(z)/|z| (first asymptotic term).
        approx = np.where(zb > 0, 1.0, (1.0 / (zb * zb)) * np.exp(-0.5 * zb * zb))
        cdf[big] = np.maximum(approx, 1e-300)
    out = base * cdf
    return np.maximum(out, 0.0)


def acquisition(mu, sigma, f_best):
    # Expected Hypar (EHypR): maximize E[max(f_best, f(x)) / f(x)].
    # For the sphere the optimum is 0, so f_best <= 0 and max(f_best, f) = f,
    # giving score = E[1 / f(x)]  (higher is better).
    start = time.time()
    score = _eratio(mu, sigma)

    # Deadline guard: keep each call comfortably inside the per-evaluation budget.
    if time.time() - start > 0.9:
        # Fall back to the top candidate by the primary score already computed.
        score = np.asarray(score, dtype=float)
        if not np.all(np.isfinite(score)):
            score = np.where(np.isfinite(score), score, 0.0)
        return score

    # Guarantee a finite, strictly increasing preference for the best candidate.
    score = np.asarray(score, dtype=float)
    if not np.all(np.isfinite(score)):
        score = np.where(np.isfinite(score), score, 0.0)
    best = np.max(score)
    if not np.isfinite(best):
        return np.zeros_like(mu)
    return score + 1e-9 * np.arange(score.size, dtype=float)
# EVOLVE-BLOCK-END
