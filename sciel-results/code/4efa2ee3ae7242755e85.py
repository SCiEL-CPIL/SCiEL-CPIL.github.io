import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function.
    
    KG(x) = E[max(f_best, f(x))] - f_best
         = f_best * (1 - Phi(z)) + mu * Phi(z) - sigma * phi(z) - f_best
         = f_best * (-Phi(z)) + mu * Phi(z) - sigma * phi(z)
         = (mu - f_best) * Phi(z) - sigma * phi(z)
    
    where z = (mu - f_best) / sigma, Phi is the standard normal CDF, and phi is the standard normal PDF.
    
    This can also be written as:
    KG(x) = (f_best - mu) * Phi(-(mu - f_best)/sigma) - sigma * phi((mu - f_best)/sigma)
    """
    start_time = time.time()
    deadline = start_time + 20.0  # Safe deadline for 2000 points
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Handle non-finite values in inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Check time
    if time.time() > deadline:
        return np.zeros_like(mu)
    
    # Compute z = (mu - f_best) / sigma
    z = (mu - f_best) / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -50, 50)
    
    # Phi(z) = CDF of standard normal
    Phi_z = norm.cdf(z)
    
    # phi(z) = PDF of standard normal
    phi_z = norm.pdf(z)
    
    # KG(x) = (mu - f_best) * Phi(z) - sigma * phi(z)
    # Note: KG is always non-negative
    kg = (mu - f_best) * Phi_z - sigma * phi_z
    
    # Ensure non-negative scores (KG should theoretically be >= 0)
    kg = np.maximum(kg, 0.0)
    
    # Ensure finite output
    kg = np.where(np.isfinite(kg), kg, 0.0)
    
    return kg
