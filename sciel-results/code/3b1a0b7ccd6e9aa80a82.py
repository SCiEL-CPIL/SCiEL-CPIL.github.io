"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per unit cost.
    # Use a small exploration bonus to encourage sampling in high-uncertainty regions
    # while still prioritizing improvement.
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Add a small exploration term proportional to sigma
    exploration = 0.1 * sigma
    return (ei + exploration) / (cost + 1e-9)
# EVOLVE-BLOCK-END
