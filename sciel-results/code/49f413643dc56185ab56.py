import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are close to the 330s limit, return a fast, safe fallback
    if time.time() - _start_time > 325:
        # Simple exploitation: prefer low mu and low cost
        safe_cost = np.maximum(cost, 1e-6)
        return -mu / safe_cost

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    safe_cost = np.maximum(cost, 1e-6)
    
    # Compute z-score for Expected Improvement
    # For minimization, improvement is f_best - y
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to prevent numerical issues in norm.cdf and norm.pdf
    z = np.clip(z, -10, 10)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Clip EI to be non-negative (mathematically it should be >= 0, but numerical errors can cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    # We use a slightly softened cost penalty (cost ** 0.8) to allow for more exploration
    # of higher-cost regions if the improvement is significant, but still heavily penalize cost.
    # The baseline uses cost ** 1.0. Inspiration 2 used cost ** 0.5 which was worse.
    # 0.8 is a middle ground that might help escape local minima in the cost landscape
    # while still respecting the budget constraint effectively.
    cost_penalty = safe_cost ** 0.8
    score = ei / cost_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
