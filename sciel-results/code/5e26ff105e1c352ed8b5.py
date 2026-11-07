"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement with a small exploration bonus
    # Use a small beta to encourage exploration when sigma is large
    beta = 0.01
    z = (f_best - mu) / (sigma + 1e-9)
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Add a small exploration term proportional to sigma
    scores = ei + beta * sigma
    return scores
# EVOLVE-BLOCK-END
