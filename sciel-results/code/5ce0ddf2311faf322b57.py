import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: (EI + Exploration Bonus) / Cost
    """
    # Deadline guard: if we are close to the limit, return a safe, fast score
    # 300s limit, safety margin 10s -> 290s
    if time.time() - _start_time > 290:
        # Return a valid score that prioritizes low cost
        # 1/cost is finite and positive
        return 1.0 / cost

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in z
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Exploration bonus: modest weight on sigma
    # This encourages visiting unexplored regions (high sigma)
    exploration_bonus = 0.5 * sigma_safe
    
    # Total benefit
    benefit = ei + exploration_bonus
    
    # Cost-aware score
    # Avoid division by zero in cost (cost >= 1 per problem statement, but safe to check)
    cost_safe = np.maximum(cost, 1e-9)
    score = benefit / cost_safe
    
    # Ensure finite outputs
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
