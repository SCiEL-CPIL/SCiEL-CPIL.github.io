"""New acquisition function using Cost-Normalized Expected Improvement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement normalized by cost.
    
    Score(x) = EI(x) / cost(x)
    
    EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    
    This approach directly optimizes the expected improvement per unit cost.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero in z calculation
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Clip z to prevent numerical instability in norm.pdf/cdf for extreme values
    # Although norm.pdf/cdf handle large values well, clipping ensures stability
    z = np.clip(z, -10, 10)
    
    # Calculate Expected Improvement
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma * phi
    
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Normalize by cost
    scores = ei / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
