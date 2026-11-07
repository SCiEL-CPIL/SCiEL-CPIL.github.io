import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 290.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling for minimization.
    
    Thompson Sampling samples from the posterior predictive distribution
    for each candidate. The candidate with the lowest sampled value
    (highest score, since we return -value) is selected.
    
    This approach provides a natural balance between exploration and
    exploitation without requiring hyperparameter tuning (like beta in UCB
    or xi in EI).
    """
    # Time guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return np.zeros_like(mu)
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Sample from the posterior N(mu, sigma^2)
    # We want to minimize the objective, so we sample y.
    # Higher score means "evaluate next", so we return -y.
    samples = np.random.normal(loc=mu, scale=sigma_safe)
    scores = -samples
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
