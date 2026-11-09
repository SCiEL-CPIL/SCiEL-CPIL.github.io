"""Rewritten acquisition for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 300.0
_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement (EI) with adaptive exploration
    for minimization.
    Returns scores where higher is better.
    """
    # Check deadline
    if time.time() > _DEADLINE - _MARGIN:
        # Return a safe fallback, pure exploitation
        return -mu
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardize for EI computation
    # For minimization, improvement is when f < f_best
    # EI = E[max(f_best - f, 0)]
    # Let Z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Standard normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure non-negative
    ei = np.maximum(ei, 0.0)
    
    # Add a small exploration bonus to escape local optima
    # Scale exploration by the relative uncertainty
    exploration_weight = 0.15
    exploration = exploration_weight * sigma_safe
    
    # Combined score: EI dominates, but exploration helps in uncertain regions
    scores = ei + exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
