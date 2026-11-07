import numpy as np
import time

_DEADLINE_MARGIN = 295.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian optimization of sphere 1d.
    
    Strategy:
    - Sample a single realization of the latent function from the posterior predictive distribution at each candidate.
    - The score is the negative of this sample.
    - The evaluator will select the candidate with the highest score (i.e., the lowest sampled value).
    - This approach naturally balances exploration (high sigma leads to wide samples) and exploitation (low mu leads to low samples) without needing to tune a kappa parameter.
    
    Parameters:
    mu: array of shape (2000,) with mean predictions
    sigma: array of shape (2000,) with standard deviation predictions
    f_best: scalar, the lowest observed value so far
    
    Returns:
    scores: array of shape (2000,) with higher values meaning more promising
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-12)
    
    # Generate random samples from the posterior predictive distribution
    # z ~ N(0, 1)
    z = np.random.standard_normal(mu.shape)
    
    # Sample: y = mu + sigma * z
    # We want to minimize y, so we maximize -y
    samples = mu + sigma * z
    scores = -samples
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time budget
    if time.time() - start_time > _DEADLINE_MARGIN:
         return np.zeros_like(mu)
         
    return scores
