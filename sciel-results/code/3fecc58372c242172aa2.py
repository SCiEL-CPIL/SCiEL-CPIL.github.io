"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Hybrid EI-LCB acquisition to balance exploitation and exploration.
    # EI component: expected improvement over f_best
    # LCB component: lower confidence bound (minimize mu - beta*sigma)
    
    # Compute EI
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute LCB (we want to minimize, so lower LCB is better)
    # For minimization: LCB = mu - beta * sigma
    # We want to maximize the score, so we negate: score_lcb = -mu + beta*sigma
    beta = 2.0
    lcb_score = -mu + beta * sigma
    
    # Normalize both components to comparable scales
    # Use simple normalization: subtract min, divide by range
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei - ei_min) / ei_range
    
    lcb_min = np.min(lcb_score)
    lcb_max = np.max(lcb_score)
    lcb_range = lcb_max - lcb_min + 1e-9
    lcb_norm = (lcb_score - lcb_min) / lcb_range
    
    # Combine with 50-50 weighting
    scores = 0.5 * ei_norm + 0.5 * lcb_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
