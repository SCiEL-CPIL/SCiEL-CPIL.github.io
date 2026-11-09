"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score for 2000 candidates.
    Uses Expected Improvement (EI) normalized by cost, with a small exploration bonus
    to avoid premature convergence in the multimodal Styblinski-Tang landscape.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    sigma = np.clip(sigma, 1e-9, None)

    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.clip(ei, 0.0, None)

    # Exploration bonus: reward high uncertainty relative to current best
    # This helps escape local optima in the multimodal landscape
    # Scale by a small factor to not overwhelm EI
    exploration = sigma * 0.1

    # Combined acquisition score
    raw_score = ei + exploration

    # Cost-aware: divide by cost to prefer cheaper evaluations
    score = raw_score / cost

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
