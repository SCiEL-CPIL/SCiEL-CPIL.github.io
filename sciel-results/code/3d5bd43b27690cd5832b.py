# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Strategy:
    - Instead of deterministic scoring (PI/LCB/EI), we sample a single realization 
      of the function from the GP posterior: y_sample = mu + sigma * z, where z ~ N(0, 1).
    - We then select the candidate that minimizes this sampled value, adjusted for cost.
    - This approach naturally balances exploration (high sigma leads to high variance 
      in the sample, potentially selecting unexplored areas) and exploitation (low sigma 
      leads to samples close to mu, refining known good areas).
    - Cost is incorporated by dividing the sampled value by the evaluation cost, 
      favoring cheaper evaluations when the expected improvement is similar.
    
    This is a structural change from the previous deterministic blend, leveraging 
    the stochastic nature of Thompson Sampling to potentially escape local optima 
    and better handle the non-uniform cost landscape.
    """
    
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT - _SAFETY_MARGIN:
        mu = np.asarray(mu, dtype=float)
        cost = np.asarray(cost, dtype=float)
        mu = np.where(np.isfinite(mu), mu, f_best)
        cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        return -mu / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma >= 0), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: Draw a single sample from the GP posterior
    # y_sample = mu + sigma * z
    # We use a fixed seed for the random draw within this call? No, we should use global random state 
    # or generate randoms. Since we are called repeatedly in a loop, using np.random is fine.
    # However, to ensure reproducibility if needed, we could use a fixed seed, but that would make 
    # the behavior deterministic over time in a weird way. Standard Thompson Sampling uses fresh randoms.
    
    # Generate standard normal randoms
    z = np.random.standard_normal(len(mu))
    
    # Sampled function values
    y_sample = mu + sigma_safe * z
    
    # Cost-aware score: We want to minimize y_sample, but penalize by cost.
    # A common approach is to minimize y_sample / cost.
    # Since the task asks for scores where HIGHER is better, we return - (y_sample / cost).
    
    # Handle division by zero or very small costs
    scores = -y_sample / (cost + 1e-9)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # If all are -inf, return a fallback (should not happen with sanitized inputs)
    if not np.any(np.isfinite(scores)):
        return -mu / (cost + 1e-9)
        
    return scores
# EVOLVE-BLOCK-END
