import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    # In GP, sigma should be > 0, but numerical errors can make it 0 or negative
    eps = 1e-9
    sigma = np.maximum(sigma, eps)
    
    # Z-score for Expected Improvement
    # Z = (f_best - mu) / sigma
    Z = (f_best - mu) / sigma
    
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Phi is norm.cdf, phi is norm.pdf
    phi_Z = norm.pdf(Z)
    Phi_Z = norm.cdf(Z)
    
    scores = (f_best - mu) * Phi_Z + sigma * phi_Z
    
    # EI is always non-negative. 
    # Due to floating point errors, it might be slightly negative. 
    # We can clip to 0, but it's not strictly necessary for ranking if we just want to maximize.
    # However, ensuring non-negativity is good practice.
    scores = np.maximum(scores, 0.0)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
