"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses a combination of LCB and EI for balanced exploration/exploitation.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Lower Confidence Bound - encourages exploration in uncertain regions
    # Beta controls exploration; for 1D Weierstrass, moderate exploration helps
    beta = 2.0
    lcb = mu - beta * sigma_safe
    
    # Normalize both to [0, 1] range for combination
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min if ei_max > ei_min else 1.0
    ei_norm = (ei - ei_min) / ei_range
    
    lcb_min, lcb_max = np.min(lcb), np.max(lcb)
    lcb_range = lcb_max - lcb_min if lcb_max > lcb_min else 1.0
    lcb_norm = (lcb - lcb_min) / lcb_range
    
    # Combine: weight EI more heavily but include LCB for exploration
    # This helps find the global minimum in 1D structured functions
    scores = 0.6 * ei_norm + 0.4 * lcb_norm
    
    # Scale back to original EI-like magnitude for consistency
    scores = scores * (ei_max - ei_min) + ei_min
    
    # Ensure all scores are finite as required by the contract
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
