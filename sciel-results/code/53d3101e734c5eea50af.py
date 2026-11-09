"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Thompson Sampling with dynamic exploration based on current best value.
    # If f_best is close to the minimum (0), we exploit (lower alpha).
    # If f_best is far from the minimum, we explore (higher alpha).
    
    # Draw standard normal noise
    eps = np.random.standard_normal(mu.shape)
    
    # Calculate dynamic alpha
    # f_best is the lowest observed value. Minimum possible is 0.
    # Maximum possible on [-5, 5] is 25.
    # Normalize f_best to [0, 1]
    f_norm = np.clip(f_best / 25.0, 0.0, 1.0)
    
    # Alpha ranges from 1.0 (pure TS) to 2.0 (more exploratory)
    # As f_best improves (decreases), f_norm decreases, alpha decreases.
    alpha = 1.0 + f_norm
    
    samples = mu + alpha * sigma * eps
    
    # Return negative samples so that higher score corresponds to lower objective
    return -samples
# EVOLVE-BLOCK-END
