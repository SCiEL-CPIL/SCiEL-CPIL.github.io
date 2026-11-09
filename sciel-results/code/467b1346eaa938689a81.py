"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Lower Confidence Bound (LCB).
    
    Uses LCB to balance exploitation (low mu) and exploration (high sigma).
    For minimization, we minimize mu - beta * sigma, so we maximize -mu + beta * sigma.
    
    mu: (n,) array of GP mean predictions
    sigma: (n,) array of GP standard deviation predictions
    f_best: current best observed value (minimum)
    
    Returns: (n,) array of scores, higher is better
    """
    # LCB parameter: controls the balance between exploration and exploitation.
    # A value of 1.5 is a standard choice for moderate exploration in BO.
    beta = 1.5
    
    # Score = -mu + beta * sigma
    # Higher score means lower expected value considering uncertainty.
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
