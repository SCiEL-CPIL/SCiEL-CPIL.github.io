"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global variables for deadline tracking
_START_TIME = None
_DEADLINE_LIMIT = 325.0  # 5 seconds margin before the 330s kill switch

def _get_elapsed_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI, uncertainty bonus, and quadratic cost penalty.
    
    The score combines:
    - Expected Improvement (EI) for exploitation
    - A small uncertainty bonus (sigma) for exploration
    - A quadratic cost penalty to strongly prefer cheap evaluations
    
    Parameters:
    mu (np.ndarray): GP posterior means, shape (N,)
    sigma (np.ndarray): GP posterior standard deviations, shape (N,)
    f_best (float): Best observed function value so far.
    cost (np.ndarray): Evaluation costs for each candidate, shape (N,)
    
    Returns:
    np.ndarray: Acquisition scores, shape (N,). Higher scores indicate
                candidates that are more likely to reduce the regret
                per unit cost.
    """
    # Deadline guard: If we are close to the time limit, return a simple, fast heuristic
    # to ensure we return valid scores and don't crash/timeout.
    # The heuristic: Maximize -mu - cost^2 (exploit known bests, prefer cheap).
    if _get_elapsed_time() > _DEADLINE_LIMIT:
        mu_arr = np.asarray(mu, dtype=np.float64)
        cost_arr = np.asarray(cost, dtype=np.float64)
        safe_scores = -mu_arr - cost_arr ** 2
        safe_scores = np.where(np.isfinite(safe_scores), safe_scores, 0.0)
        return safe_scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-zero
    sigma_safe = np.maximum(sigma, 1e-8)
    # Ensure cost is non-zero
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is standard normal CDF and phi is standard normal PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Small uncertainty bonus to encourage some exploration
    # Using a slightly higher coefficient than before to balance with quadratic penalty
    uncertainty_bonus = 0.02 * sigma_safe
    
    # Quadratic cost penalty: strongly penalize expensive evaluations
    # Cost range is [1, 2] (1 + mean(x) where x in [0,1])
    # cost^2 ranges [1, 4], so the penalty is significant
    # Tuning the penalty coefficient: 0.5 was used before, let's try 0.4 to allow more exploration
    cost_penalty = 0.4 * (cost_safe ** 2)
    
    # Combine: EI + uncertainty bonus - cost penalty
    scores = ei + uncertainty_bonus - cost_penalty
    
    # Replace any non-finite values with 0.0 to ensure validity
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
