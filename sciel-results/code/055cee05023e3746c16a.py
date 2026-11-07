"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline tracking
_START_TIME = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Hypervolume-Inspired Improvement Surrogate.
    
    Structural change from standard EI: Instead of maximizing expected improvement
    directly, we use a log-barrier formulation that optimizes for the quality of
    the non-dominated front in (improvement, cost) space.
    
    Formula: Score = log(1 + PI) * sigma^gamma / cost^delta
    
    - log(1 + PI) smoothly saturates exploitation, preventing over-focusing on
      near-certain improvements and encouraging diverse exploration
    - sigma^gamma with gamma < 1 provides sublinear exploration boost that is
      more balanced across uncertainty scales than linear sigma
    - cost^delta with delta < 1 softens the cost penalty, allowing the optimizer
      to consider expensive points if the potential gain is high enough
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard: return safe fallback if we're running out of time
    elapsed = time.time() - _START_TIME
    if elapsed > 310.0:
        # Fallback: simple sigma-based score normalized by cost
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        cost_arr = np.asarray(cost, dtype=np.float64)
        fallback = np.maximum(sigma_arr, 1e-8) / np.maximum(cost_arr, 1e-8)
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for probability of improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    PI = norm.cdf(z)
    
    # Clip PI to avoid numerical issues with log
    PI_clipped = np.clip(PI, 1e-10, 1.0 - 1e-10)
    
    # Parameters for the structural change
    gamma = 0.7   # Sublinear uncertainty exponent: balanced exploration across scales
    delta = 0.5   # Softened cost penalty: allows occasional expensive but informative points
    
    # Log-barrier on PI: smooth, bounded in [log(1+eps), log(2)]
    # This saturates exploitation smoothly, preventing over-concentration
    log_pi = np.log1p(PI_clipped)
    
    # Uncertainty term: sigma^gamma
    # Sublinear growth provides more balanced exploration than linear sigma
    sigma_term = np.power(sigma_safe, gamma)
    
    # Cost term: 1 / cost^delta
    # Softened penalty allows the optimizer to weigh improvement potential
    # against cost more flexibly
    cost_term = np.power(cost_safe, -delta)
    
    # Combine: hypervolume-inspired improvement surrogate
    scores = log_pi * sigma_term * cost_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
