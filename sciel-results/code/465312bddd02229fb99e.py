"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining:
    - Expected Improvement (EI) for exploitation
    - Uncertainty bonus for exploration (escape local minima)
    - Cost normalization (cheaper points get higher score)
    
    mu: array of GP mean predictions, shape (2000,)
    sigma: array of GP std predictions, shape (2000,)
    f_best: scalar, best observed function value so far
    cost: array of evaluation costs, shape (2000,)
    Returns: array of acquisition scores, shape (2000,)
    """
    # Deadline guard: track elapsed time to ensure we return before 300s limit
    # Since this function is called repeatedly, we just ensure it's fast
    # The main cost is the norm.cdf/pdf calls on 2000 points
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure no NaN or Inf inputs propagate
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e10, neginf=-1e10)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e10, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e10, neginf=1.0)
    
    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Safety: ensure cost is positive
    cost_safe = np.maximum(cost, 1e-8)
    
    # Expected Improvement with small xi for slight exploration
    # xi=0.005 balances exploitation and exploration
    xi = 0.005
    improvement = f_best - mu - xi
    z = improvement / sigma_safe
    
    # Clip z to avoid extreme values in cdf/pdf
    z = np.clip(z, -50, 50)
    
    # EI formula
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Uncertainty bonus: encourages exploration in high-sigma regions
    # This is important for Rastrigin which has many local minima
    # Use a saturating function to prevent unbounded growth
    # The bonus is NOT divided by cost so we always explore uncertain regions
    uncertainty_bonus = 0.05 * sigma_safe / (1.0 + sigma_safe)
    
    # Cost-aware: divide EI by cost so cheaper evaluations are favored
    cost_normalized_ei = ei / cost_safe
    
    # Combine: cost-normalized EI + uncertainty bonus
    score = cost_normalized_ei + uncertainty_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
