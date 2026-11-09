"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected improvement with exponential cost penalty.
    # Cost is in [1, 2], so exp(-cost) ranges from ~0.135 to ~0.368.
    # This exponentially suppresses high-cost regions more aggressively than 1/cost,
    # which better aligns with the Levy 2D landscape where the minimum is near the center.
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return ei * np.exp(-cost)
# EVOLVE-BLOCK-END
