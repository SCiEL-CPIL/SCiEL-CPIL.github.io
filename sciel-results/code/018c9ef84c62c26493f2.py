"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Use a hybrid acquisition: UCB with adaptive beta + EI-like shaping
    # to balance exploration and exploitation in the short-horizon Ackley landscape.
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-10)
    
    # Adaptive beta: larger when few points evaluated (high uncertainty), smaller later
    # Since we don't know n directly, use the scale of sigma to modulate
    # For 2000 candidates, use a moderate exploration constant
    beta = 1.5
    
    # UCB score
    ucb = mu + beta * sigma
    
    # EI component for exploitation pressure
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize both to [0, 1] range for stable combination
    ucb_min, ucb_max = ucb.min(), ucb.max()
    ei_min, ei_max = ei.min(), ei.max()
    
    ucb_range = ucb_max - ucb_min + 1e-10
    ei_range = ei_max - ei_min + 1e-10
    
    ucb_norm = (ucb - ucb_min) / ucb_range
    ei_norm = (ei - ei_min) / ei_range
    
    # Blend: weight toward UCB for exploration, EI for exploitation
    # Use 0.6 UCB + 0.4 EI to favor exploration in early stages
    score = 0.6 * ucb_norm + 0.4 * ei_norm
    
    # Return higher scores for more promising points
    return score
# EVOLVE-BLOCK-END
