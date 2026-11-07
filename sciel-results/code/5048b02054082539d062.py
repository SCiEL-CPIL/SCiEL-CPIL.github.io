"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT_LIMIT = 320.0

def acquisition(mu, sigma, f_best):
    # Check time limit
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to simple UCB if time is running out
        return -mu + 0.5 * sigma
    
    # Use Expected Improvement (EI) which scored 1.0.
    # To improve exploration in multimodal landscapes, we add a small
    # exploration noise to the standard deviation. This prevents the
    # acquisition function from becoming too greedy in regions where
    # the GP uncertainty is very low but the function might still be
    # far from the global minimum.
    
    # Small additive noise to sigma to encourage exploration
    exploration_noise = 0.05
    sigma_eff = sigma + exploration_noise
    
    z = (f_best - mu) / sigma_eff
    # Handle potential numerical issues
    z = np.clip(z, -10, 10)
    
    scores = (f_best - mu) * norm.cdf(z) + sigma_eff * norm.pdf(z)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
