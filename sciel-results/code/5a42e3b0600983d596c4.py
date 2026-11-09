"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses a hybrid of Expected Improvement and an uncertainty bonus.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    # Clip z to prevent numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Add a small exploration bonus based on uncertainty
    # This helps escape local minima and explore high-uncertainty regions
    # The weight is small to maintain exploitation focus
    exploration_weight = 0.05
    sigma_max = np.max(sigma_safe)
    if sigma_max > 0:
        bonus = exploration_weight * (sigma_safe / sigma_max)
    else:
        bonus = np.zeros_like(sigma_safe)
    
    scores = ei + bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
