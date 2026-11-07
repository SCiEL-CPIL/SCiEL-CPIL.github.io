# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for Thompson Sampling
_ts_state = {'seed': 0}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Draws a single sample from the GP posterior and selects the candidate
    that minimizes this sample divided by its cost. This treats the objective
    as a stochastic reward-to-go ratio, balancing exploration and exploitation
    through the natural randomness of the posterior sample.
    """
    # Ensure numerical stability
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Update the seed for reproducible but varying samples
    _ts_state['seed'] += 1
    rng = np.random.RandomState(_ts_state['seed'])
    
    # Draw a single random sample from the GP posterior
    # f_sample(x) ~ N(mu(x), sigma(x)^2)
    z = rng.standard_normal(len(mu))
    f_sample = mu - sigma_safe * z
    
    # Ensure cost is positive to avoid division by zero or negative costs
    cost_safe = np.maximum(cost, eps)
    
    # Thompson Sampling score: minimize f_sample / cost
    # We want to select the point with the best (lowest) f_sample / cost ratio
    # Since the acquisition function returns scores where higher = evaluate next,
    # we use -f_sample / cost_safe
    scores = -f_sample / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
