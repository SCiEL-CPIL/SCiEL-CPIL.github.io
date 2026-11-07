import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation at candidate points
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    # Knowledge Gradient: expected improvement of the minimum
    # E[min(f_best, f(x))] where f(x) ~ N(mu(x), sigma(x))
    
    # For each candidate x, the expected value of min(f_best, f(x)) is:
    # E[min(f_best, f(x))] = mu(x) - sigma(x) * phi(d) + f_best * Phi(-d)
    # where d = (f_best - mu(x)) / sigma(x)
    # But we want to maximize the expected reduction in the minimum, which is:
    # KG(x) = f_best - E[min(f_best, f(x))]
    #       = f_best - [mu(x) - sigma(x) * phi(d) + f_best * Phi(-d)]
    #       = f_best - mu(x) + sigma(x) * phi(d) - f_best * Phi(-d)
    #       = (f_best - mu(x)) * (1 - Phi(-d)) + sigma(x) * phi(d)
    #       = (f_best - mu(x)) * Phi(d) + sigma(x) * phi(d)
    
    # Handle cases where sigma is very small to avoid division by zero
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute d = (f_best - mu) / sigma
    d = (f_best - mu) / sigma_safe
    
    # Compute phi(d) and Phi(d)
    phi_d = norm.pdf(d)
    Phi_d = norm.cdf(d)
    
    # KG score: expected reduction in the observed minimum
    # Higher KG means we expect to improve f_best more by evaluating here
    kg = (f_best - mu) * Phi_d + sigma_safe * phi_d
    
    # Ensure finite scores
    kg = np.where(np.isfinite(kg), kg, 0.0)
    
    return kg
