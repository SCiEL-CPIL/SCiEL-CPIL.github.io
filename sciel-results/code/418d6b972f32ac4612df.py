"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with a small
    UCB-style exploration bonus to handle multimodal landscapes.
    EI provides rank-based exploration pressure, while the bonus helps
    escape local optima in regions of high uncertainty.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Basic safety checks
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        return np.zeros_like(mu)
    
    # Handle cases where sigma is very small
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -50, 50)
    
    # Compute Phi(z) and phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Add a small exploration bonus to encourage exploring uncertain regions
    # This helps escape local optima in the acquisition landscape
    # The bonus is scaled by sigma to be proportional to uncertainty
    exploration_bonus = 0.01 * sigma_safe
    
    scores = ei + exploration_bonus
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
