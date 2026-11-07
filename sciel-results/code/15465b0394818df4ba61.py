# EVOLVE-BLOCK-START
import numpy as np
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        sigma_safe = np.maximum(sigma, 1e-9)
        # Fallback: Simple UCB
        return -mu + 1.5 * sigma_safe
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Structural Change: Independent Thompson Sampling (Noisy UCB)
    # Instead of a deterministic score like EI or UCB, we sample from the
    # posterior predictive distribution at each candidate point independently.
    # This mimics sampling a function from the GP posterior and selecting
    # the point with the lowest sampled value.
    # Score(x) = - (mu(x) + sigma(x) * z_x)
    # where z_x ~ N(0, 1)
    
    # Generate standard normal noise for each candidate
    z = np.random.standard_normal(n)
    
    # Compute sampled function values
    # We want to minimize f, so we maximize -f_sampled
    f_sampled = mu + sigma_safe * z
    scores = -f_sampled
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
