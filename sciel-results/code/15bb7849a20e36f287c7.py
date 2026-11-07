"""Rewritten program for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Parameters:
    -----------
    mu : array_like, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : array_like, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed function value so far (not used directly in score, but part of context).
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    start_time = time.time()
    deadline = start_time + 280.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    if len(mu) == 0:
        return np.array([])
        
    # Generate standard normal noise for Thompson Sampling
    # This consumes random numbers from the global RNG, which is expected for TS
    z = np.random.randn(len(mu))
    
    # Sample from the posterior predictive distribution: f(x) ~ N(mu(x), sigma(x)^2)
    samples = mu + sigma * z
    
    # We want to minimize the function value, so we maximize the negative of the sample
    scores = -samples
    
    # Ensure finite values to avoid invalid selection
    # If a sample is inf, it should not be selected. -inf is the lowest score.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check deadline (though this operation is very fast)
    if time.time() > deadline:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
