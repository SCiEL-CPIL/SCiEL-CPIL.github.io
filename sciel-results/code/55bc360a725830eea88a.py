"""Thompson Sampling acquisition function for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for minimization.
    Samples a single realization from the Gaussian posterior and returns
    negative sampled values as scores (higher = more promising).
    
    Args:
        mu: array of shape (2000,) - posterior mean in objective units
        sigma: array of shape (2000,) - posterior standard deviation in objective units
        f_best: float - lowest observed value
    
    Returns:
        array of shape (2000,) - scores where higher means evaluate next
    """
    global _start_time
    
    # Deadline guard: if we're running out of time, fall back to simple greedy
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return negative mu (greedy exploitation) to avoid timeout
        return -mu
    
    # Thompson Sampling: sample a single realization from the posterior
    # For each candidate i, f_i ~ N(mu_i, sigma_i^2)
    # We want to minimize f, so we sample and pick the one with lowest f
    # Score = -sampled_value, so higher score = lower sampled f = more promising
    
    # Generate random samples from the Gaussian posterior
    # Use numpy's random generator for efficiency
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(size=mu.shape)
    
    # Score is negative of the sampled value (higher score = lower f = better)
    scores = -samples
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
