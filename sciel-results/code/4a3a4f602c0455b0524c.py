"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected improvement with a small exploration term to encourage exploration
    # when uncertainty is high.
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small UCB term to balance exploration and exploitation
    beta = 0.5  # Exploration-exploitation trade-off parameter
    ucb = -mu + beta * sigma
    
    # Combine EI and UCB
    scores = 0.7 * ei + 0.3 * ucb
    
    return scores
# EVOLVE-BLOCK-END
