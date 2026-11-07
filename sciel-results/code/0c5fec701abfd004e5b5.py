"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : ndarray of shape (2000,)
        Mean of GP predictions.
    sigma : ndarray of shape (2000,)
        Standard deviation of GP predictions.
    f_best : float
        Best observed value so far.
    cost : ndarray of shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray of shape (2000,)
        Acquisition scores (higher = more promising).
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware EI: normalize by cost
    cost_safe = np.maximum(cost, 1e-9)
    cost_aware_ei = ei / cost_safe
    
    # Infer number of evaluations from total cost
    # Start with 5 uniform random points, each costs 1 + mean(x_scaled)
    # Total budget is 30, so we can estimate how many evaluations have been done
    # Average cost per evaluation is roughly 1.5 (since x is uniform in [0,1]^4, mean ~0.5)
    # So roughly total_cost / 1.5 evaluations have been made
    # But we don't have total_cost directly, so use a heuristic based on f_best
    # Early on: f_best is poor (high value), explore more
    # Late on: f_best is good (low value), exploit more
    
    # Use a proxy: the magnitude of improvement potential
    # If f_best is very high, we're early -> more exploration
    # If f_best is low, we're late -> more exploitation
    
    # Shekel 4d minimum is -10.5364, typical values range from -10 to 0+
    # Normalize f_best to get a sense of progress
    # f_best starts around 0 or positive, ends near -10.5
    progress = np.clip((f_best - (-10.5364)) / (0 - (-10.5364)), 0, 1)
    # progress = 1 early, 0 late
    
    # Adaptive weights: more exploration early, more exploitation late
    # exploration_weight: 0.5 -> 0.1 as progress goes 1 -> 0
    exploration_weight = 0.1 + 0.4 * progress
    exploitation_weight = 1.0 - exploration_weight
    
    # Combine cost-aware EI and exploration bonus with adaptive weights
    exploration_bonus = exploration_weight * sigma_safe / cost_safe
    scores = exploitation_weight * cost_aware_ei + exploration_bonus
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
