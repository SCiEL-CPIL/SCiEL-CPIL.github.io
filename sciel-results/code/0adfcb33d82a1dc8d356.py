"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Upper Confidence Bound per unit cost.
    # UCB encourages exploration (high sigma) and exploitation (low mu).
    # Dividing by cost makes it cost-aware.
    beta = 2.0
    # Clip sigma to avoid division by zero or numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    # For minimization, we want to maximize (f_best - mu + beta * sigma)
    # If mu < f_best, (f_best - mu) is positive.
    # If mu > f_best, (f_best - mu) is negative.
    scores = (f_best - mu + beta * sigma_safe) / cost
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, -1e10)
    return scores
# EVOLVE-BLOCK-END
