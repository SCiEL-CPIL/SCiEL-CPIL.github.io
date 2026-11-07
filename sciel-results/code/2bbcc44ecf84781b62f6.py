# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = 325.0
_START_TIME = time.time()

def _check_deadline():
    """Return True if we should abort and return current best."""
    return (time.time() - _START_TIME) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Strategy:
    - Instead of computing Expected Improvement (EI), we sample a single scalar value
      from the Gaussian posterior for each candidate.
    - The score is the negative of this sample divided by the cost.
    - This naturally balances exploration (high sigma leads to larger variance in samples,
      occasionally yielding very low values) and exploitation (low sigma concentrates samples
      around the mean).
    - Dividing by cost encourages finding low values in cheap regions.
    
    This is a structural change from the deterministic EI - penalty approach to a 
    stochastic sampling approach.
    """
    # Check deadline first
    if _check_deadline():
        # Return a neutral score if we're running out of time
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-6, 1e6)
    # Cost is 1 + mean(x_scaled), so it's in [1, 2] for x in [-5, 5] scaled to [0, 1]
    # But let's be safe and clip to a reasonable range
    cost = np.clip(cost, 0.5, 10.0)
    
    # Thompson Sampling:
    # Sample one value from N(mu, sigma) for each candidate
    # We generate a single random standard normal vector
    eps = np.random.standard_normal(len(mu))
    
    # The sample is mu + sigma * eps
    # We want to minimize the function, so we prefer lower values.
    # Score = - (mu + sigma * eps) / cost
    # Higher score is better.
    
    samples = mu + sigma * eps
    
    # Calculate score
    # To avoid division by zero or negative costs (though cost should be positive),
    # we use np.where or just divide since cost is clipped to >= 0.5
    score = -samples / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # If there are -inf values, replace with a very low finite number
    # to ensure the function returns finite scores as required by the contract
    # "Return 2000 finite scores"
    min_score = np.min(score)
    if not np.isfinite(min_score):
        # This shouldn't happen if we handled inf properly, but just in case
        score = np.where(np.isfinite(score), score, -1e6)
    else:
        # Replace any remaining non-finite values (should be none)
        score = np.where(np.isfinite(score), score, min_score - 1.0)
        
    return score
# EVOLVE-BLOCK-END
