"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level time tracking for deadline guard
_module_start = time.time()
_DEADLINE = 320.0  # Safety margin before 330s hard limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using geometric-mean balancing
    of normalized UCB and inverse cost, treating cost as a discount factor
    in a stochastic shortest-path framing.
    
    Args:
        mu (np.ndarray): Mean of GP posterior, shape (N,).
        sigma (np.ndarray): Std dev of GP posterior, shape (N,).
        f_best (float): Best observed function value.
        cost (np.ndarray): Cost of evaluating each candidate, shape (N,).
        
    Returns:
        np.ndarray: Acquisition scores, shape (N,). Higher is better.
    """
    # Deadline guard
    if time.time() - _module_start > _DEADLINE:
        return np.random.uniform(0, 1, size=len(mu))
    
    N = len(mu)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB for minimization: we want to minimize, so lower mu is better.
    # UCB value = mu - beta * sigma (lower is better for the objective)
    # But for acquisition score (higher = evaluate next), we use:
    # score_ucb = f_best - mu + beta * sigma
    # This gives high scores when mu is low (exploitation) or sigma is high (exploration)
    
    # Adaptive beta based on the scale of the problem
    # Use a moderate exploration constant
    beta = 1.5
    
    ucb_score = f_best - mu + beta * sigma_safe
    
    # Normalize UCB scores to [0, 1] range
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min
    if ucb_range < 1e-9:
        ucb_norm = np.ones_like(ucb_score)
    else:
        ucb_norm = (ucb_score - ucb_min) / ucb_range
        ucb_norm = np.maximum(ucb_norm, 1e-9)
    
    # Normalize cost to [0, 1] range
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range < 1e-9:
        cost_norm = np.ones_like(cost)
    else:
        cost_norm = (cost - cost_min) / cost_range
        cost_norm = np.clip(cost_norm, 1e-9, 1.0)
    
    # Inverse cost: lower cost is better, so 1/cost_norm gives higher weight to cheap points
    # But we need to be careful: cost_norm is in [1e-9, 1], so 1/cost_norm is in [1, 1e9]
    # Let's use a softer inverse: (1 + epsilon) / (cost_norm + epsilon)
    eps = 0.1
    inv_cost = (1.0 + eps) / (cost_norm + eps)
    
    # Normalize inv_cost to [0, 1]
    inv_min = np.min(inv_cost)
    inv_max = np.max(inv_cost)
    inv_range = inv_max - inv_min
    if inv_range < 1e-9:
        inv_cost_norm = np.ones_like(inv_cost)
    else:
        inv_cost_norm = (inv_cost - inv_min) / inv_range
        inv_cost_norm = np.maximum(inv_cost_norm, 1e-9)
    
    # Geometric mean of normalized UCB and inverse cost
    # This balances exploitation/exploration (UCB) with cost efficiency (inv_cost)
    # Geometric mean is more stable than division
    scores = np.sqrt(ucb_norm * inv_cost_norm)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
