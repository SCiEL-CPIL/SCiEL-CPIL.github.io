import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Computes the Shifted Expected Improvement (EI) acquisition function.
    
    Parameters:
    mu : np.ndarray
        Mean of the Gaussian Process posterior, shape (N,)
    sigma : np.ndarray
        Standard deviation of the Gaussian Process posterior, shape (N,)
    f_best : float
        Best observed value (minimum) so far.
        
    Returns:
    np.ndarray
        Acquisition scores for each candidate, shape (N,)
    """
    # Use a negative xi to increase exploration pressure.
    # This shifts the improvement threshold lower, encouraging the search
    # to explore regions with high uncertainty even if the expected value
    # is not immediately superior to f_best.
    xi = -0.2
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu - xi) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    scores = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
