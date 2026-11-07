"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Lower Confidence Bound (LCB).
    
    For minimization, we maximize the score:
    Score = -mu + beta * sigma
    
    beta is adapted based on the current best value f_best.
    Larger f_best implies we are far from the optimum, so we explore more (higher beta).
    Smaller f_best implies we are close, so we exploit more (lower beta).
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Ensure f_best is a scalar float
    f_best_val = float(f_best)
    
    # Calculate adaptive beta
    # The +1e-9 prevents sqrt(0) issues, though f_best >= 0 for Beale.
    beta = 1.5 * np.sqrt(f_best_val + 1e-9)
    
    # Compute LCB scores
    # We want to minimize f(x), so we look for low mu.
    # We want to explore, so we look for high sigma.
    # Score = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure finite outputs
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
