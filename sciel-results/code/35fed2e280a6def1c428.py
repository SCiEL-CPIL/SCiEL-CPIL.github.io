"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track call count to adapt exploration
_call_count = 0
_call_time = 0.0


def acquisition(mu, sigma, f_best):
    global _call_count, _call_time
    
    _call_count += 1
    _call_time = time.time()
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    
    # Ensure mu is finite
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    
    # Compute z for EI
    z = (f_best - mu_safe) / sigma_safe
    
    # Clip z to avoid extreme values causing numerical issues
    z = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small UCB exploration term to encourage exploration
    # beta decreases as we get more evaluations
    beta = 0.5 * np.exp(-_call_count / 10.0)
    ucb_term = beta * sigma_safe
    
    # Combine EI and UCB
    scores = ei + ucb_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
