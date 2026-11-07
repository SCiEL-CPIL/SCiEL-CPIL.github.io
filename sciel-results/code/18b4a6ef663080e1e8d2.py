import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    # Higher EI means higher expected improvement over the current best.
    
    # Ensure numerical stability
    # sigma should be >= 0, but clamp to avoid division by zero or negative std
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # Phi is the CDF of the standard normal distribution
    # phi is the PDF of the standard normal distribution
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI is non-negative by definition (since f_best - mu can be negative, 
    # but the term is 0 when mu > f_best because z < 0 implies Phi(z) is small 
    # and the terms cancel out appropriately? 
    # Actually, standard EI formula:
    # If mu > f_best, z < 0. 
    # (f_best - mu) is negative. Phi(z) is small positive. 
    # sigma * pdf(z) is positive.
    # It is possible for EI to be slightly negative due to floating point or 
    # if mu is very close to f_best from above? 
    # Standard EI is defined as max(0, EI). Let's clip to 0 to be safe.
    ei = np.maximum(ei, 0.0)
    
    return ei
