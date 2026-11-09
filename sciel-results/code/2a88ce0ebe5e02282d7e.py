"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guard
_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Computes the acquisition scores for a given set of candidate points.
    
    This implementation uses Expected Improvement (EI) for minimization.
    EI naturally balances exploration and exploitation and is robust for
    small-budget Bayesian optimization on rugged landscapes like Ackley.
    """
    # Deadline guard: 330s limit. Return immediately if close to limit.
    if time.time() - _START_TIME > 320:
        return np.zeros_like(mu)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the CDF and phi is the PDF of the standard normal.
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-10)
    
    # Improvement: f_best - mu (positive when mu < f_best)
    improvement = f_best - mu
    
    # Standardized improvement
    z = improvement / sigma
    
    # EI = improvement * Phi(z) + sigma * phi(z)
    # For large negative z, EI approaches 0 (no improvement expected)
    # For large positive z, EI approaches improvement (high improvement expected)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    scores = improvement * Phi_z + sigma * phi_z
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
