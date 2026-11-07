"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a stage-adaptive blend of LCB (exploration) and EI (exploitation).
    
    Args:
        mu: (N,) array of GP mean predictions.
        sigma: (N,) array of GP standard deviations.
        f_best: float, best observed value so far.
        cost: (N,) array of costs for each candidate.
    
    Returns:
        (N,) array of acquisition scores (higher is better).
    """
    N = len(mu)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Ensure cost is positive to avoid division by zero
    cost_safe = np.maximum(cost, 1e-10)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei_term = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei_term = np.maximum(ei_term, 0.0)
    
    # Lower Confidence Bound (LCB) score
    # We minimize mu - beta*sigma, so we maximize -(mu - beta*sigma)
    beta = 2.0
    lcb_score = -mu + beta * sigma_safe
    
    # Stage-adaptive blending
    # alpha = 0 -> LCB (Exploration), alpha = 1 -> EI (Exploitation)
    # We want to explore when f_best is high (far from optimum 0).
    # We want to exploit when f_best is low (close to optimum 0).
    # Therefore, alpha should be low when f_best is high, and high when f_best is low.
    # alpha = sigmoid((threshold - f_best) / scale)
    
    threshold = 0.5
    scale = 0.2
    
    # Calculate alpha
    # If f_best is very large, (threshold - f_best) is very negative, exp is large, alpha is small.
    # If f_best is very small, (threshold - f_best) is positive, exp is small, alpha is large.
    alpha = 1.0 / (1.0 + np.exp(-(threshold - f_best) / scale))
    
    # Combine scores
    raw_scores = (1.0 - alpha) * lcb_score + alpha * ei_term
    
    # Cost-aware: divide by cost
    scores = raw_scores / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
