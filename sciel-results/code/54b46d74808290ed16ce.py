import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Z = (f_best - mu) / sigma
    
    # Clip sigma to avoid division by zero and numerical instability
    # A small epsilon ensures stability.
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate Z
    Z = (f_best - mu) / sigma_safe
    
    # Calculate Phi(Z) and phi(Z)
    # norm.cdf is Phi, norm.pdf is phi
    Phi_Z = norm.cdf(Z)
    phi_Z = norm.pdf(Z)
    
    # Calculate EI
    scores = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
