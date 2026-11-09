import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    Samples a value from the posterior for each candidate and returns the negative
    of the sample as the score (higher is better).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive for sampling
    safe_sigma = np.where(sigma < 1e-12, 1e-12, sigma)
    
    # Sample from the posterior: N(mu, sigma^2)
    # We use the global numpy random state
    samples = np.random.normal(mu, safe_sigma)
    
    # For minimization, we want the lowest value.
    # So the score should be high for low values.
    scores = -samples
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    return scores
