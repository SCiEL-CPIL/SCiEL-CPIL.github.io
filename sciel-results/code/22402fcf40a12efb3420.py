"""Rewritten acquisition for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Minimum-variance acquisition: score candidates solely by posterior std dev.
    This forces exploration of unexplored regions, preventing premature convergence.
    """
    # Ensure finite output
    sigma = np.asarray(sigma, dtype=np.float64)
    # Higher sigma = more uncertain = higher score (explore more)
    scores = sigma.copy()
    # Clamp any non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END
