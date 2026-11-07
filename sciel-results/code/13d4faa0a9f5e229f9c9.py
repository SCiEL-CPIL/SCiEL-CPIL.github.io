"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Gaussian-process-based Expected Improvement
    (EI) with a small, adaptive exploration term.

    The exploration weight is tuned to be small relative to typical EI values,
    so that exploitation dominates but exploration can still break ties in
    uncertain regions.  A hard time guard ensures we always return within the
    evaluator's 300 s budget.
    """
    # ----- time guard (safety margin of ~5 s) -----
    t0 = time.time()
    deadline = t0 + 295.0  # leave 5 s for caller overhead

    n = mu.shape[0]

    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)

    # z-score for the standard normal CDF/PDF
    z = (f_best - mu) / sigma_safe

    # Expected Improvement for minimization:
    #   EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # We compute phi(z) via the standard normal PDF.
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Adaptive exploration weight:
    #   - scale relative to the spread of EI so the bonus is meaningful
    #   - cap it so it never overwhelms exploitation
    ei_scale = np.max(np.abs(ei)) + 1e-12
    w = 0.05 * ei_scale / (np.max(sigma_safe) + 1e-12)
    w = min(w, 0.3 * ei_scale)  # hard cap: exploration ≤ 30 % of EI scale

    scores = ei + w * sigma_safe

    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # If we are dangerously close to the deadline, return immediately
    if time.time() > deadline - 0.5:
        return scores

    # Optional: a second pass with a slightly different weight to break
    # near-ties (only if we have time to spare).
    if time.time() < deadline - 2.0:
        w2 = 0.02 * ei_scale / (np.max(sigma_safe) + 1e-12)
        w2 = min(w2, 0.15 * ei_scale)
        scores2 = ei + w2 * sigma_safe
        scores2 = np.where(np.isfinite(scores2), scores2, 0.0)

        # Blend: prefer the score set that gives a wider spread (more
        # discriminative), which tends to pick more distinct candidates.
        spread1 = np.ptp(scores)
        spread2 = np.ptp(scores2)
        if spread2 > spread1:
            scores = scores2

    return scores
# EVOLVE-BLOCK-END
