# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function.
    
    mu: array of shape (n,), posterior mean at candidates
    sigma: array of shape (n,), posterior std at candidates
    f_best: float, best observed value so far
    
    Returns: array of shape (n,), higher means evaluate next.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate improvement: f_best - mu
    # We want to minimize, so improvement is how much lower the value could be
    imp = f_best - mu
    
    # z-score
    z = imp / sigma_safe
    
    # EI = imp * Phi(z) + sigma * phi(z)
    # Phi is CDF, phi is PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    scores = imp * Phi_z + sigma_safe * phi_z
    
    # Ensure non-negative (EI should theoretically be >= 0)
    scores = np.maximum(scores, 0.0)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
