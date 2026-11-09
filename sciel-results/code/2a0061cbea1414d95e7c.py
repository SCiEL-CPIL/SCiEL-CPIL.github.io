"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    normalized by evaluation cost.
    
    EI is generally superior to PI because it accounts for the magnitude
    of potential improvement, not just the probability. This leads to
    more directed searches towards the global minimum.
    
    Score = EI(x) / cost(x)
    
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    """
    # Avoid division by zero in z calculation
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # If f_best < mu, EI should be 0 (handled by max later, but formula works out)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Numerical stability: EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    scores = ei / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
