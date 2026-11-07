import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Computes the Expected Improvement (EI) acquisition score for minimization.
    
    Args:
        mu: Mean of the GP posterior (shape (2000,))
        sigma: Standard deviation of the GP posterior (shape (2000,))
        f_best: Best observed value so far (scalar)
        
    Returns:
        Scores (shape (2000,)), higher is better for selection.
    """
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # Avoid division by zero
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # For minimization, EI is defined as:
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) 
    #        + sigma(x) * phi((f_best - mu(x))/sigma(x))
    # where Phi is the CDF of the standard normal and phi is the PDF.
    
    # If sigma is very small, we can approximate EI as max(0, f_best - mu)
    # But the formula above handles this naturally if we're careful.
    
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    # For numerical stability, when z is very negative, Phi(z) ~ 0 and phi(z) ~ 0, so EI ~ 0
    # When z is very positive, Phi(z) ~ 1 and phi(z) ~ 0, so EI ~ f_best - mu
    
    # Use scipy.stats.norm for accuracy
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (it should be by definition, but numerical errors can occur)
    ei = np.maximum(ei, 0.0)
    
    # Ensure finiteness
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
