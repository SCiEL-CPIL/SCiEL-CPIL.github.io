"""Rewritten cost-aware acquisition for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: use a blend of Expected Improvement (EI) and
    Probability of Improvement (PI) weighted by cost, with a time guard.
    """
    t0 = time.time()
    deadline = t0 + 290.0  # safety margin before 300s limit

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)

    # z-score for improvement
    z = (f_best - mu) / sigma_safe

    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Probability of Improvement
    pi = norm.cdf(z)

    # Cost: evaluating x costs 1 + mean(x scaled to [0,1])
    # Since we don't have x coordinates, we use the provided cost array
    # The cost is already provided, so we normalize it
    cost_safe = np.maximum(cost, 1e-9)

    # Cost-aware: divide by cost to get value per unit cost
    # Blend EI and PI: EI captures magnitude of improvement, PI captures likelihood
    # Weight: use EI primarily, but add a small PI component to encourage exploration
    # when EI is small but PI is high (uncertain regions)
    
    # Normalize EI and PI for blending
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    pi_max = np.max(pi) if np.max(pi) > 0 else 1.0
    
    # Use a weighted combination: 0.8 * EI + 0.2 * PI, both normalized
    blended = 0.8 * (ei / ei_max) + 0.2 * (pi / pi_max)
    
    # Cost-aware score
    score = blended / cost_safe

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    # Time guard: if we're close to deadline, return immediately
    if time.time() > deadline - 5.0:
        return score

    return score
# EVOLVE-BLOCK-END
