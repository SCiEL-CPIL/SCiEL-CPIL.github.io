import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization using Expected Improvement (EI).
    
    This function computes the Expected Improvement of evaluating a new point 
    compared to the current best value. For minimization, EI is defined as:
    EI(x) = E[max(f_best - f(x), 0)]
    
    Given f(x) ~ N(mu(x), sigma(x)^2), the closed-form solution is:
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    Phi is the standard normal CDF and phi is the standard normal PDF.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    # A small epsilon is used for numerical stability
    epsilon = 1e-10
    sigma_safe = np.maximum(sigma, epsilon)
    
    # Compute the standardized distance z
    z = (f_best - mu) / sigma_safe
    
    # Compute Expected Improvement
    # Phi(z) = norm.cdf(z)
    # phi(z) = norm.pdf(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability might cause tiny negative values)
    ei = np.maximum(ei, 0.0)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
