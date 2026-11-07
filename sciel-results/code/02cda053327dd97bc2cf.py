"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement (EI) acquisition function.
    # We use a very weak cost penalty to ensure the search is not biased
    # away from the global minimum, which may reside in a higher cost region.
    # The global minimum of the 3-hump camel function is significantly lower
    # than local minima, so EI naturally favors it. A strong cost penalty
    # can cause the optimizer to settle for a cheaper local minimum.
    
    # Calculate z-score for EI
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost penalty
    # cost is in [1, 2]. 
    # We use a very small linear penalty to slightly discourage expensive points
    # without overriding the improvement signal.
    # Slope 0.05 results in a max penalty factor of 1.05.
    cost_penalty = 1.0 + 0.05 * (cost - 1.0)
    
    # To further encourage exploration in unexplored regions (where sigma is high),
    # EI already does this. However, if we want to be more aggressive, we could
    # add a small bonus for high sigma. But EI is usually sufficient.
    
    scores = ei / cost_penalty
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
