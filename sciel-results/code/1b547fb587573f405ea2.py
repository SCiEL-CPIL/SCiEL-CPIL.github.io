"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) approach.
    # We want to minimize f(x). LCB estimates a lower bound on the true function value.
    # LCB(x) = mu(x) - beta * sigma(x)
    # To select the point with the lowest LCB (most promising minimum),
    # we maximize the negative LCB: Score = -mu(x) + beta * sigma(x).
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Beta controls the trade-off between exploitation (low mu) and exploration (high sigma).
    # A value around 1.0 to 2.0 is common. 
    # Since we have a limited number of evaluations (30 total), 
    # a moderate exploration is needed to find the global minimum at (3, 0.5).
    beta = 1.5
    
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
