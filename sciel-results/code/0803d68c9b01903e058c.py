"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add an exploration bonus to encourage visiting uncertain regions
    # This helps escape local minima in multimodal functions like Branin
    exploration = 1.0 * sigma
    
    return ei + exploration
# EVOLVE-BLOCK-END
