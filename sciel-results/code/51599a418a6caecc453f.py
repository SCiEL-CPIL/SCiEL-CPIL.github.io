import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Pure Thompson Sampling.
    
    This approach samples from the posterior predictive distribution to select the next point.
    This stochastic nature helps escape local minima in multimodal landscapes like Michalewicz.
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior, shape (N,)
    sigma (np.ndarray): Standard deviation of the GP posterior, shape (N,)
    f_best (float): The best (lowest) objective value observed so far.
    
    Returns:
    np.ndarray: Scores for each candidate, shape (N,). Higher scores are preferred.
    """
    # Check deadline
    if time.time() - _START_TIME > _TIME_LIMIT - 5.0:
        # Return a safe fallback if we are about to timeout
        # This shouldn't happen in normal operation but prevents crash
        return np.zeros_like(mu)

    # Thompson Sampling: Sample from the posterior predictive distribution
    # f(x) ~ N(mu(x), sigma(x)^2)
    # We want to minimize f(x), so we maximize -f(x).
    # Sample z ~ N(0, 1)
    z = np.random.standard_normal(mu.shape)
    
    # The sample from the posterior is mu + sigma * z
    # The score is the negative of this sample (since higher score is better for selection, and we want low objective values)
    score = -(mu + sigma * z)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
