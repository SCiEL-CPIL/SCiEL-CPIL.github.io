"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: pure UCB with adaptive exploration for multimodal minimization.
    
    Uses Upper Confidence Bound (UCB) which has shown strong performance on
    multimodal functions like Michalewicz 2D. The exploration parameter is
    adapted based on the current uncertainty level to balance exploration
    and exploitation.
    """
    # Track time for safety
    t_start = time.time()
    
    # Ensure shapes and types
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=-1e-6)
    sigma = np.maximum(sigma, 1e-12)
    
    n = len(mu)
    
    # Adaptive UCB: use a beta that balances exploration and exploitation
    # For minimization, we want to maximize: score = -mu + beta * sigma
    # Higher beta favors exploration, lower beta favors exploitation
    
    # Base beta for UCB
    beta_base = 2.0
    
    # Adapt beta based on uncertainty: when overall uncertainty is high,
    # we can afford more exploration; when low, exploit more
    sigma_median = np.median(sigma)
    
    # Normalize sigma to get a relative measure of uncertainty
    # If sigma_median is small relative to the spread of mu, we're in exploitation mode
    mu_range = np.ptp(mu) if n > 1 else 1.0
    
    if mu_range > 1e-12:
        # Relative uncertainty: how large is sigma compared to the range of mu
        rel_uncertainty = sigma_median / (mu_range + 1e-12)
    else:
        rel_uncertainty = 1.0
    
    # Scale beta: when rel_uncertainty is high, use larger beta (more exploration)
    # When rel_uncertainty is low, use smaller beta (more exploitation)
    # Clamp to reasonable range [1.0, 3.0]
    beta = np.clip(beta_base * (0.5 + rel_uncertainty), 1.0, 3.0)
    
    # UCB score for minimization: higher is better
    score = -mu + beta * sigma
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Safety: check time
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:  # Safety margin before 300s limit
        # Return a simple fallback
        return -mu
    
    return score
# EVOLVE-BLOCK-END
