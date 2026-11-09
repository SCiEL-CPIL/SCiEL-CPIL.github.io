import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores using Expected Improvement (EI) for minimization.
    
    EI is defined as the expected amount by which the objective function 
    could be improved from the current best observed value f_best.
    
    For minimization:
    EI(x) = (f_best - mu(x)) * Phi(Z) + sigma(x) * phi(Z)
    where Z = (f_best - mu(x)) / sigma(x)
    Phi is the standard normal CDF and phi is the standard normal PDF.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-12)
    
    # Calculate the Z-score
    # If mu is very close to f_best, Z is large or small depending on sign
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement
    # Term 1: (f_best - mu) * Phi(Z)
    # Term 2: sigma * phi(Z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EI should be non-negative, but due to numerical errors it might be slightly negative
    # Clip to 0
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite values
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)
    
    return ei
