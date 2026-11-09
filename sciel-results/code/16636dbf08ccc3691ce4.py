import numpy as np
import time

# Global state to track time and potentially seed the RNG for reproducibility
# within a single run if needed, though the task implies stateless calls.
# We use a global RNG instance to ensure consistent behavior across calls
# in the same process, but reset it? No, the task says "for each seed 0, 1 and 2".
# The acquisition function is called 25 times per seed (30 evals - 5 initial).
# If we use a global RNG, the sequence of random numbers will depend on the order of calls.
# This is standard for Thompson Sampling in BO loops.

_rng = np.random.default_rng()

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive to avoid division issues or log issues if we used them
    sigma = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: Sample one value per candidate from the posterior
    # sample_i ~ N(mu_i, sigma_i^2)
    # We want to minimize, so we want the lowest sample.
    # The acquisition function must return scores where HIGHER is better.
    # So we return -sample.
    
    # Generate random numbers
    z = _rng.standard_normal(size=mu.shape)
    samples = mu + sigma * z
    
    # The score is the negative of the sampled value
    # Higher score means lower sampled value, which is what we want to pick.
    score = -samples
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
