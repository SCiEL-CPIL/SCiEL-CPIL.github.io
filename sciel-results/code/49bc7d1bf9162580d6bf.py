"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import ndtr

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    
    Uses Adaptive UCB: score = mu + beta * sigma
    where beta adapts based on f_best to balance exploration/exploitation.
    For minimization, lower mu + beta*sigma is better, so we return negative
    for ranking (higher score = evaluate next).
    
    Actually, for minimization UCB should be: minimize mu - beta*sigma
    So we return -(mu - beta*sigma) = -mu + beta*sigma for max-ranking.
    
    But wait - the standard approach for minimization is to use:
    score = -mu + beta * sigma (maximize this to find good points)
    
    Adaptive beta: start high for exploration, decrease as f_best improves.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: simple EI
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        phi_z = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
        ei = (f_best - mu) * ndtr(z) + sigma_safe * phi_z
        return np.where(np.isfinite(ei), ei, 0.0)

    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Adaptive beta: scale with sqrt(log(f_best + 1)) for early exploration
    # When f_best is large (early), beta is larger -> more exploration
    # When f_best approaches 0, beta shrinks -> more exploitation
    log_scale = np.sqrt(np.log(f_best + 1.0 + 1e-10))
    beta = 2.0 * log_scale + 0.1
    
    # UCB score for minimization: we want to minimize mu - beta*sigma
    # Return negative so higher = better for ranking
    score = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
