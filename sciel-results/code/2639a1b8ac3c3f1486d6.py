"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Hybrid acquisition: blend of Expected Improvement and Upper Confidence Bound
    # EI focuses on exploitation, UCB on exploration
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    
    # Upper Confidence Bound (for minimization: lower is better)
    # We want to maximize (f_best - mu + beta * sigma) to find potentially better values
    beta = 0.5
    ucb = f_best - mu + beta * sigma
    
    # Normalize both to [0, 1] range for blending
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    
    if ei_max - ei_min > 1e-10:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    if ucb_max - ucb_min > 1e-10:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Blend: 70% EI, 30% UCB
    score = 0.7 * ei_norm + 0.3 * ucb_norm
    
    return score
# EVOLVE-BLOCK-END
