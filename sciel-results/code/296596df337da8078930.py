"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on griewank 2d.
    Uses a blend of Expected Improvement and Probability of Improvement,
    with an exponential cost penalty that sharply prefers cheap points.
    """
    t0 = time.time()

    # Ensure inputs are valid
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Compute z-score
    z = (f_best - mu) / sigma_safe

    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Probability of Improvement
    pi = norm.cdf(z)

    # Blend EI and PI: EI captures magnitude of improvement, PI captures likelihood
    # Use a weighted combination that emphasizes PI more as we get closer to optimum
    # f_best for griewank starts around 0.01-0.1 and goes to ~0
    # As f_best decreases, PI becomes more informative
    progress = np.clip(f_best, 0, 1)
    # Weight: more PI weight when f_best is small (late stage)
    pi_weight = 0.5 + 0.5 * (1.0 - progress)  # ranges from 0.5 to 1.0

    # Normalize EI to [0,1]-ish scale for blending
    ei_max = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    ei_norm = ei / ei_max

    # Combined acquisition
    base = pi_weight * pi + (1.0 - pi_weight) * ei_norm

    # Exponential cost penalty: strongly penalize expensive points
    # Cost ranges from 1 to 2 (1 + mean(x scaled to [0,1]))
    # Use a penalty that makes cost=2 about e^(-lambda) times cost=1
    # lambda controls strength; ~1.5 gives ratio of ~0.22
    lambda_cost = 1.5
    cost_penalty = np.exp(-lambda_cost * (cost_safe - 1.0))

    score = base * cost_penalty

    # Clamp non-finite values
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline guard
    if time.time() - t0 > 280:
        return score

    return score
# EVOLVE-BLOCK-END
