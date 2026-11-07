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
    # Ensure sigma is non-zero to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the improvement over the best
    improvement = f_best - mu
    
    # Calculate the z-score
    z = improvement / sigma_safe
    
    # Calculate the expected improvement
    # EI = improvement * Phi(z) + sigma * phi(z)
    # norm.cdf(z) is Phi(z)
    # norm.pdf(z) is phi(z)
    scores = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite and non-negative (EI should be >= 0)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
