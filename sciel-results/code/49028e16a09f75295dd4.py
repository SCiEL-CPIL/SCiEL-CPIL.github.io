"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization.
    # We want to minimize the function, so we look for points with low mu.
    # We add a term proportional to sigma to encourage exploration.
    # Score = -mu + beta * sigma. Maximizing this score selects the next point.
    beta = 1.5
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    score = -mu + beta * sigma
    return score
# EVOLVE-BLOCK-END
