"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Expected Improvement (EI) for minimization.
    mu, sigma, cost: shape (2000,)
    f_best: scalar
    Returns: shape (2000,) finite scores, higher = evaluate next.
    """
    start_time = time.time()
    deadline = 300.0
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Calculate Z score for Expected Improvement
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Phi is the CDF of the standard normal distribution
    # phi is the PDF of the standard normal distribution
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Cost-aware score: EI / cost
    # We want to maximize the expected improvement per unit cost
    score = ei / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return score
        
    return score
# EVOLVE-BLOCK-END
