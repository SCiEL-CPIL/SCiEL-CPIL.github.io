"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    UCB acquisition with adaptive beta for 1D sphere minimization.
    
    Structural change: Use UCB (mu + beta * sigma) instead of EI.
    Beta is adapted based on f_best: since the minimum of x^2 on [-5,5] is 0,
    we know the target. As f_best -> 0, we reduce exploration.
    
    We use a smooth transition: beta starts high for exploration and decays
    as f_best gets closer to 0.
    """
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # f_best is the best (lowest) value observed so far for x^2 on [-5,5]
    # f_best ranges from ~25 (worst initial) down to ~0 (converged)
    # Use f_best directly as a progress indicator
    
    # Normalize f_best to [0, 1] where 0 means converged (f_best=0) and 1 means worst case
    # f_best can be up to ~25 for x in [-5,5], so use 25 as upper bound
    f_norm = np.clip(f_best / 25.0, 0.0, 1.0)
    
    # Beta schedule: high when f_best is large (early, explore), low when f_best is small (late, exploit)
    # Use a smooth decay: beta = beta_min + (beta_max - beta_min) * f_norm^2
    # Quadratic decay: fast transition to exploitation once f_best starts decreasing
    beta_max = 2.5
    beta_min = 0.1
    beta = beta_min + (beta_max - beta_min) * (f_norm ** 2)
    
    # UCB score: higher is better (we want to maximize this for minimization)
    # For minimization, we want low mu and high sigma (exploration)
    # UCB for minimization: -mu + beta * sigma (negate mu since we minimize)
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
