"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson-Sampling-like acquisition:
    score = -mu - beta * (sigma / sigma_ref)
    where beta is scaled by f_best to shift from exploration to exploitation.
    sigma_ref is a quantile-based normalizer.
    """
    # Guard against degenerate sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Quantile-normalized sigma: use 95th percentile as reference scale
    # This makes the exploration term dimensionless and robust
    sigma_ref = np.percentile(sigma_safe, 95)
    if sigma_ref < 1e-12:
        sigma_ref = 1.0
    
    normalized_sigma = sigma_safe / sigma_ref
    
    # Dynamic exploration weight:
    # When f_best is large (early), explore more aggressively.
    # When f_best approaches 0, focus on exploitation.
    # Use a smooth transition: beta = 1.5 * (f_best / (f_best + 0.1))
    # At f_best = 0.1, beta = 0.75; at f_best = 1.0, beta ~ 1.36; at f_best = 10, beta ~ 1.49
    # This keeps beta in a reasonable range while shifting balance.
    beta = 1.5 * (f_best / (f_best + 0.1))
    
    # Score: higher is better. We want to minimize, so -mu is the exploitation term.
    # The exploration term rewards high normalized sigma.
    score = -mu - beta * normalized_sigma
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
