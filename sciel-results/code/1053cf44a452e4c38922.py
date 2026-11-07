"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Minimum Variance acquisition function.
    # Scores candidates solely by their posterior standard deviation.
    # This forces the optimizer to explore regions with high uncertainty,
    # preventing premature convergence and resolving GP uncertainty.
    
    # Ensure sigma is finite and non-negative.
    # If sigma is negative or non-finite, treat it as 0 (or a small epsilon if we want to avoid exact 0s, 
    # but 0 is a valid score for "no uncertainty").
    scores = np.where(np.isfinite(sigma) & (sigma >= 0), sigma, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
