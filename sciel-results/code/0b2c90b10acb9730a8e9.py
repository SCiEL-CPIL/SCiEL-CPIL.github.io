# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 330s kill


def _time_remaining():
    return 330.0 - (time.time() - _START_TIME) - _DEADLINE_MARGIN


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement
    beyond a dynamically relaxed threshold, divided by evaluation cost.
    
    Score = Phi((f_best - epsilon - mu) / sigma) / cost
    where epsilon = 0.05 (relaxed threshold for meaningful gain).
    
    This prioritizes cheap evaluations that have high confidence of
    beating the current best by at least epsilon.
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Dynamically relaxed threshold: we want improvement of at least epsilon
    epsilon = 0.05
    threshold = f_best - epsilon
    
    # Compute z-score for probability of improvement beyond threshold
    z = (threshold - mu) / sigma
    
    # Probability of improvement: Phi(z)
    # Clip z to avoid numerical issues
    z = np.clip(z, -50.0, 50.0)
    prob_improve = norm.cdf(z)
    
    # Cost-effective score: probability divided by cost
    score = prob_improve / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
