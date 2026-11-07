import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses Thompson Sampling to balance exploration and exploitation.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    start_time = time.time()
    deadline = 325.0

    # Ensure sigma is strictly positive to avoid division by zero or NaNs
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Generate standard normal samples
    # We sample y ~ N(mu, sigma^2)
    # y = mu + sigma * z
    z = np.random.randn(mu.shape[0])
    
    # Sampled objective values
    sampled_values = mu + sigma_safe * z
    
    # We want to minimize the objective.
    # The caller selects the candidate with the HIGHEST score.
    # Therefore, we return the negative of the sampled values.
    # A lower sampled value (better) results in a higher (less negative) score.
    scores = -sampled_values
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check deadline (though TS is very fast, good practice)
    if time.time() - start_time > deadline:
        return scores

    return scores
