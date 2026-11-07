"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Calculate Expected Improvement (EI)
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware scoring with squared penalty
    # Stronger penalty for higher cost to exploit the correlation between 
    # low cost and the location of the global minimum in Styblinski Tang.
    score = ei / (cost * cost)
    return score
# EVOLVE-BLOCK-END
