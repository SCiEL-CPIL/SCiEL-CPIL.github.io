import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Look-Ahead Expected Improvement (LAEI) approximation.
    Uses Expected Improvement (EI) as a proxy for the lookahead value,
    which directly estimates the expected reduction in the global minimum
    (regret) based on the local GP posterior.
    """
    # Ensure sigma is positive and avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Z-score for minimization
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    scores = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # EI is theoretically non-negative
    scores = np.maximum(scores, 0.0)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
