"""Rewritten acquisition for cost-aware Bayesian optimization of Rastrigin 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PI)
    with an exploration component, divided by evaluation cost.
    
    The PI-based approach is more robust than EI for multimodal functions like
    Rastrigin because it doesn't over-weight high-variance regions that promise
    large but unlikely improvements.
    
    mu: array of GP mean predictions, shape (2000,)
    sigma: array of GP std predictions, shape (2000,)
    f_best: scalar, best observed function value so far
    cost: array of evaluation costs, shape (2000,)
    Returns: array of acquisition scores, shape (2000,)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Probability of Improvement: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This is the probability that evaluating x will yield a better value than f_best
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Add a small exploration bonus based on uncertainty
    # Use a saturating function to keep it bounded
    # The exploration term encourages visiting under-explored regions
    exploration = 0.05 * sigma_safe / (1.0 + sigma_safe)
    
    # Combine PI and exploration
    score_base = pi + exploration
    
    # Cost-aware: divide by cost so cheaper evaluations are favored
    cost_safe = np.maximum(cost, 1e-8)
    score = score_base / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
