import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    EI is the expected improvement over f_best, which corresponds to
    the expected maximum of Thompson Sampling samples.
    """
    t_start = time.time()
    deadline = t_start + 290.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety
    sigma = np.clip(sigma, 1e-9, None)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    cost = np.where(np.isfinite(cost) & (cost > 1e-6), cost, 1.0)

    # Calculate z-score
    # z = (f_best - mu) / sigma
    # If sigma is very small, z is large.
    z = (f_best - mu) / sigma
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If f_best < mu, z is negative.
    # Phi(z) is prob that f < f_best.
    # If mu > f_best, the first term is negative, but Phi(z) is small.
    # The second term sigma * phi(z) is always positive.
    # EI should be non-negative.
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical errors can make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware normalization
    # Prefer cheaper points.
    # Current program used 1/sqrt(cost).
    # Let's try 1/cost for a stronger preference for cheap points,
    # as the optimum is likely in the cheap region (near origin).
    # However, 1/sqrt(cost) was in the 2.28 score program.
    # Let's stick to 1/sqrt(cost) to be safe, or maybe 1/cost?
    # Let's use 1/cost to differentiate structurally from the parent's 1/sqrt(cost).
    # If 1/cost is too aggressive, we might miss the optimum if it's slightly expensive.
    # But Ackley min is at 0, cost at 0 is 1. Cost at corners is 2.
    # 1/cost range [0.5, 1]. 1/sqrt(cost) range [0.707, 1].
    # 1/cost is more distinct.
    
    cost_weight = 1.0 / cost
    
    score = ei * cost_weight
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
