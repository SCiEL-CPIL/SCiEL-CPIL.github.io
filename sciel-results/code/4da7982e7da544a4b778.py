import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed function value so far (unused in pure TS, but kept for signature).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising.
    """
    # Sanitize sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    sigma_safe = np.nan_to_num(sigma_safe, nan=1e-12, posinf=1e6, neginf=1e-12)
    
    # Generate random samples from the posterior distribution
    # y ~ N(mu, sigma^2)
    z = np.random.standard_normal(size=mu.shape)
    samples = mu + sigma_safe * z
    
    # We want to minimize the objective.
    # A lower sampled value is better.
    # We return the negative of the sample so that higher score = lower value.
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
