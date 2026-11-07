"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function for Bayesian optimization.
    
    Uses standard Expected Improvement (EI) for minimization, normalized by cost.
    EI measures the expected improvement over f_best in objective units.
    Dividing by cost gives cost-aware EI (higher score = evaluate next).
    """
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Standardized improvement: how many std devs below f_best is mu?
    # z = (f_best - mu) / sigma
    # When mu < f_best, z > 0 (improvement expected)
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical overflow in norm.cdf/norm.pdf
    z = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # Cost is in [1, 2] range (1 + mean of x scaled to [0,1])
    scores = ei / cost
    
    # Ensure finite outputs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
