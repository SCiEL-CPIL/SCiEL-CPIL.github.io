"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global time tracking for deadline guard
_ACQ_START_TIME = time.time()
_ACQ_TIME_LIMIT = 330.0
_ACQ_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    with a small additive cost penalty.
    
    Maximizes EI - lambda * cost, where:
    - EI = (f_best - mu) * Phi(z) + sigma * phi(z) for minimization
    - z = (f_best - mu) / sigma
    - lambda is a small constant (0.1) to prefer cheaper evaluations
      when EI is similar, conserving the total budget.
    """
    # Deadline guard
    if time.time() - _ACQ_START_TIME > _ACQ_TIME_LIMIT - _ACQ_SAFETY_MARGIN:
        # Return a neutral score to allow the loop to finish safely
        return np.zeros_like(mu)

    # Ensure inputs are proper arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Small additive cost penalty to prefer cheaper evaluations
    # when predicted improvement is similar
    lambda_cost = 0.1
    scores = ei - lambda_cost * cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
