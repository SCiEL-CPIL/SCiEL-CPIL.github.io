"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 285

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (POI) with a
    small UCB-style exploration term. POI prioritizes regions where there is
    a high probability of beating the current best, which can escape local
    optima in multimodal landscapes like Branin differently than EI.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Basic safety checks
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        return np.zeros_like(mu)
    
    # Handle cases where sigma is very small
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Probability of Improvement for minimization:
    # POI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -50, 50)
    
    # Compute Phi(z)
    poi = norm.cdf(z)
    
    # Add a small UCB-style exploration bonus to encourage exploring uncertain regions
    # This helps find new basins in the multimodal Branin landscape
    # Scale the bonus relative to the range of mu to be adaptive
    mu_range = np.ptp(mu) if np.ptp(mu) > 1e-10 else 1.0
    exploration_beta = 0.05
    exploration_bonus = exploration_beta * sigma_safe / mu_range
    
    scores = poi + exploration_bonus
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
