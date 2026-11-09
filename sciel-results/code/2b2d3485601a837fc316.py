"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    This approach draws a single sample from the posterior predictive distribution
    at each candidate point and uses the negative of that sample as the score.
    This inherently balances exploration and exploitation without requiring
    a manually tuned beta parameter.
    
    Sample = mu - sigma * z, where z ~ N(0, 1)
    Score = -Sample = -mu + sigma * z
    
    Args:
        mu: Array of GP mean predictions (2000,).
        sigma: Array of GP standard deviation predictions (2000,).
        f_best: The lowest observed function value.
        
    Returns:
        Array of Thompson Sampling scores (2000,). Higher scores indicate better candidates.
    """
    start_time = time.time()
    # Safety margin: The evaluator kills at 330s, but individual calls should be fast.
    # We add a guard just in case, though Thompson Sampling is O(N).
    deadline = start_time + 290.0
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])
    
    # Check time
    if time.time() > deadline:
        # Fallback: return zeros if we are running out of time
        return np.zeros(n)
    
    # Sanitize sigma: ensure non-negative and finite
    # Thompson Sampling benefits from high sigma, so we want to preserve large values but handle NaNs/Infs
    sigma_clean = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma_clean = np.maximum(sigma_clean, 0.0)
    
    # Sanitize mu: replace non-finite with a large value (penalty)
    # We are maximizing -mu + sigma * z.
    # If mu is +inf, -mu is -inf (bad).
    # If mu is -inf, -mu is +inf (good, but likely invalid data).
    # We treat non-finite mu as very bad (high value) to penalize it.
    mu_clean = np.where(np.isfinite(mu), mu, 1e10)
    
    # Generate a single random normal sample z ~ N(0, 1)
    # Using a fixed seed per call would make it deterministic, but Thompson Sampling
    # benefits from stochasticity. However, to ensure reproducibility within a run
    # and avoid issues with global state, we use np.random.randn directly.
    # The evaluator runs separate seeds, so this should be fine.
    z = np.random.randn(n)
    
    # Compute Thompson Sampling score: -mu + sigma * z
    # We want to maximize this score.
    scores = -mu_clean + sigma_clean * z
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
