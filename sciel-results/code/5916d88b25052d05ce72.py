import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient acquisition function (1-step KG is equivalent to Expected Improvement).
    
    We use the analytical closed-form solution for Expected Improvement (EI)
    for a Gaussian posterior N(mu, sigma^2).
    EI(x) = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    This is more stable and accurate than Monte Carlo approximation.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Calculate the CDF and PDF of the standard normal distribution
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure non-negative (numerical errors might cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    return ei
