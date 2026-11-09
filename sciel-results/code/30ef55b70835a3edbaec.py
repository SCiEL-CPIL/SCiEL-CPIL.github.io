# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Probability of Improvement (PoI)
    with an uncertainty bonus.
    
    Score = PoI / cost + exploration_weight * sigma / scale
    
    Parameters:
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviation predictions
    f_best: float - lowest observed objective value
    cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for PoI (minimization)
    # PoI = P(mu < f_best) = Phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -30, 30)
    
    # Probability of Improvement
    poi = norm.cdf(z)
    
    # Cost-adjusted PoI: prioritize candidates that offer high likelihood of success per unit cost
    poi_cost = poi / cost_safe
    
    # Adaptive scale for uncertainty bonus
    # Use the gap to the known minimum (0) to scale exploration
    # If f_best is large, we are far from the optimum, so explore more
    gap = np.maximum(f_best, 1e-6)
    scale = np.sqrt(gap)
    
    # Uncertainty bonus: encourages visiting high-uncertainty regions
    # Scaled relative to the current best value
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * sigma_safe / scale
    
    # Combined score: higher is better
    scores = poi_cost + exploration_bonus
    
    # Handle any non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t_start
    if elapsed > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
