"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of:
    1. Expected Improvement (EI) - standard exploration/exploitation balance
    2. Probability of Improvement (PI) - more aggressive exploitation
    3. Cost normalization - accounts for evaluation cost
    
    The score is higher for candidates that are likely to improve f_best
    while having lower cost.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement (PI) - how likely this point is to beat f_best
    pi = norm.cdf(z)
    
    # Cost-adjusted score:
    # - EI / cost: expected gain per unit cost
    # - Add a small PI term to break ties and encourage exploration of promising regions
    # - Use a blended approach: primarily EI/cost, with PI as a secondary signal
    
    # Primary: EI normalized by cost
    ei_per_cost = ei / (cost + 1e-10)
    
    # Secondary: PI (probability this point improves f_best), lightly weighted
    # This helps in early iterations when EI is noisy
    pi_weighted = 0.1 * pi
    
    # Combine: use EI/cost as primary, PI as tie-breaker
    # The 0.1 weight on PI is small enough not to dominate but helps distinguish
    # between points with similar EI/cost
    score = ei_per_cost + pi_weighted
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
