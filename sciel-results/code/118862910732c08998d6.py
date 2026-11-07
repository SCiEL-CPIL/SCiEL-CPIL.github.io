# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using the product of Probability of Improvement (PI)
    and Expected Improvement (EI), normalized by cost.
    
    Score = (PI * EI) / cost
    
    This metric strongly favors points that are both likely to be better than the current best
    (high PI) and have a large expected gain (high EI), while penalizing expensive evaluations.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * pi + sigma_safe * norm.pdf(z)
    
    # Ensure non-negative
    ei = np.maximum(ei, 0.0)
    pi = np.clip(pi, 0.0, 1.0)
    
    # Combined Score
    # PI * EI emphasizes exploitation of likely good points with high potential.
    score = (pi * ei) / cost
    
    # Ensure scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
