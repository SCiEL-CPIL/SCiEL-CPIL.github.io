import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    EI is a standard acquisition function for Bayesian optimization that
    balances exploration and exploitation by computing the expected value
    of improvement over the current best observation.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the improvement: f_best - mu (positive means better than best)
    improvement = f_best - mu
    
    # Z-score
    z = improvement / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Compute Phi(z) and phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Calculate EI
    ei = improvement * Phi_z + sigma_safe * phi_z
    
    # Ensure non-negative EI (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite values
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)
    
    return ei
