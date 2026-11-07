"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    start = time.time()
    deadline = 290.0

    # Safety margins
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)

    EI = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)

    # Cost-aware EI
    scores = EI / cost_safe

    # Add a UCB-like exploration bonus to balance exploitation and exploration
    # Using a larger coefficient to encourage more exploration in uncertain regions
    beta = 0.5
    ucb_bonus = beta * sigma_safe / cost_safe
    scores = scores + ucb_bonus

    # Check deadline before returning
    if time.time() - start > deadline:
        return np.where(np.isfinite(-mu / cost_safe), -mu / cost_safe, 0.0)

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
