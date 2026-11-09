"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We maximize -mu + kappa * sigma.
    # Adaptive kappa: start with high exploration, reduce as f_best improves.
    # f_best is the current minimum.
    # If f_best is large, we are far from optimum, explore more.
    # If f_best is small, we are close, exploit more.
    
    # Ensure f_best is non-negative for the heuristic, though it should be for sphere.
    fb = max(f_best, 0.0)
    
    # kappa = base + scale * f_best
    # base = 2.0 is a common starting point for UCB.
    # scale = 5.0 amplifies exploration when f_best is large.
    kappa = 2.0 + 5.0 * fb
    
    # Calculate UCB score
    # sigma is standard deviation
    score = -mu + kappa * sigma
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
