"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (PI) acquisition function for minimization.
    
    PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    We maximize PI / cost to get cost-aware scores.
    
    Args:
        mu: Array of shape (N,) with predicted means.
        sigma: Array of shape (N,) with predicted standard deviations.
        f_best: Scalar, the best observed value so far.
        cost: Array of shape (N,) with evaluation costs.
        
    Returns:
        Array of shape (N,) with acquisition scores (higher is better).
    """
    # Compute z-score for PI
    epsilon = 1e-8
    sigma_safe = np.maximum(sigma, epsilon)
    
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement: Phi(z)
    pi_scores = norm.cdf(z)
    
    # Make cost-aware by dividing by cost
    cost_safe = cost + epsilon
    
    scores = pi_scores / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
