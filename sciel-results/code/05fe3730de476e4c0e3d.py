# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on rastrigin 2d.
    
    Uses a combination of Probability of Improvement (PI) and Expected Improvement (EI),
    weighted by cost, to balance exploration of cheap regions with exploitation of
    promising low-value areas.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to non-negative
    ei = np.maximum(ei, 0.0)
    
    # Combine PI and EI with a balance parameter
    # PI encourages exploration of uncertain regions, EI targets high-improvement regions
    # Weight them: 0.7 * PI + 0.3 * EI (scaled)
    # Scale EI by a factor relative to typical improvement magnitudes
    ei_scale = np.max(ei) + 1e-9
    ei_normalized = ei / ei_scale
    
    # Combine
    score = 0.7 * pi + 0.3 * ei_normalized
    
    # Apply cost penalty: divide by cost to prefer cheaper evaluations
    # Use a soft penalty: score / (cost + epsilon)
    cost_safe = np.maximum(cost, 1e-9)
    score = score / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
