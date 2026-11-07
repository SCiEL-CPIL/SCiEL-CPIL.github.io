"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Probability of Improvement with soft cost penalty.
    
    PI = Phi((f_best - mu) / sigma)
    Score = PI^(1/cost)
    
    This naturally saturates at 1.0 for certain improvements and decays
    smoothly with uncertainty. The exponentiation by 1/cost penalizes
    expensive evaluations without the division instability of EI/cost.
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # Clip to avoid log(0) issues; PI should be in (0, 1)
    pi = np.clip(pi, 1e-10, 1.0 - 1e-10)
    
    # Soft cost penalty: raise to power 1/cost
    # Higher cost -> smaller exponent -> more penalty for uncertainty
    # Lower cost -> larger exponent -> preserves high PI values
    # cost is in [1, 2] since cost = 1 + mean(x_scaled) where x_scaled in [0,1]
    cost_safe = np.maximum(cost, 1e-9)
    
    # log-space computation for numerical stability
    log_score = np.log(pi) / cost_safe
    score = np.exp(log_score)
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
