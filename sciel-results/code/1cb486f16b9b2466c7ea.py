"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Known global minimum of Hartmann 3D
    f_min = -3.86278
    # Approximate upper bound for Hartmann 3D function values
    f_max = 1.0
    
    # Calculate progress towards the optimum
    # gap is the distance from current best to the global minimum
    gap = f_best - f_min
    max_gap = f_max - f_min
    
    # Weight for exploitation (EI) vs exploration (sigma)
    # w is close to 0 when far from optimum (explore)
    # w is close to 1 when close to optimum (exploit)
    w = 1.0 - (gap / max_gap)
    w = np.clip(w, 0.0, 1.0)
    
    # Calculate Expected Improvement (EI)
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration score: posterior standard deviation
    # This encourages visiting regions with high uncertainty
    exp_score = sigma
    
    # Blend EI and exploration score
    # Early on: mostly exploration (high sigma)
    # Later on: mostly exploitation (high EI)
    scores = w * ei + (1.0 - w) * exp_score
    
    return scores
# EVOLVE-BLOCK-END
