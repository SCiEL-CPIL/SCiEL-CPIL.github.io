"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement per unit cost
    # EI for minimization: E[max(f_best - f(x), 0)]
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Then divide by cost to get cost-aware EI
    
    # Handle near-zero sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: EI per unit cost
    # cost is in [1, 2], so dividing by cost gives higher score to cheaper points
    acquisition_scores = ei / cost
    
    # Ensure finite values
    acquisition_scores = np.where(np.isfinite(acquisition_scores), acquisition_scores, -np.inf)
    
    return acquisition_scores
# EVOLVE-BLOCK-END
