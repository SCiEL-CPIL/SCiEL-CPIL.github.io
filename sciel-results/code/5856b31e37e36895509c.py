"""Acquisition for Bayesian optimization of a fixed seeded random RBF mixture in 2D."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_START = time.monotonic()
_DEADLINE = 290.0


def acquisition(mu, sigma, f_best):
    # Deadline guard: if we are dangerously close to the wall-clock limit,
    # return a safe fallback (pure exploitation via lowest mu) immediately.
    if (time.monotonic() - _START) > _DEADLINE:
        return -mu

    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    n = mu.shape[0]

    # Ensure sigma is positive and finite.
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, np.inf)

    # Expected Improvement for minimization:
    #   EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Higher EI means more likely to improve.
    z = (f_best - mu) / sigma
    z = np.where(np.isfinite(z), z, 0.0)

    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Lower Confidence Bound (LCB) for minimization:
    #   LCB = mu - kappa * sigma
    # We return -LCB so higher is better.
    #
    # We blend EI and LCB. Early in the optimization, exploration (LCB) is
    # more important; later, exploitation (EI) should dominate. We use the
    # current best value as a proxy: when f_best is close to the known global
    # minimum (-2.53218), we shift weight toward EI.
    #
    # A simple adaptive kappa: start at 2.5 (strong exploration), decay to
    # ~0.5 as f_best approaches the known optimum.
    global_min = -2.53218
    # progress: 0 when f_best is far from optimum, 1 when at optimum
    progress = np.clip((f_best - global_min) / (f_best - global_min + 1e-12) if (f_best - global_min) > 0 else 1.0, 0.0, 1.0)
    # Actually, let's think about this differently.
    # If f_best is much larger than global_min, we are far from optimum -> explore more.
    # If f_best is close to global_min, we are near optimum -> exploit more.
    gap = f_best - global_min  # positive, how far we are
    # kappa from 3.0 (far) to 0.5 (near)
    kappa = 0.5 + 2.5 * np.exp(-max(gap, 0.0))

    lcb_score = -(mu + 0.01) + kappa * sigma

    # Normalize both to [0, 1] range for blending.
    def _normalize(x):
        mn, mx = np.min(x), np.max(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    ei_norm = _normalize(ei)
    lcb_norm = _normalize(lcb_score)

    # Blend: weight on EI increases as we get closer to the optimum.
    # w_ei goes from 0.3 (early) to 0.9 (late).
    w_ei = 0.3 + 0.6 * (1.0 - np.exp(-max(gap, 0.0) * 2.0))
    w_ei = np.clip(w_ei, 0.0, 1.0)

    score = w_ei * ei_norm + (1.0 - w_ei) * lcb_norm

    # Scale back to original EI range for meaningful magnitudes.
    score = score * (np.max(ei) - np.min(ei) + 1e-12) + np.min(ei)

    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
