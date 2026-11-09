"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement per unit cost
    with a PSI-inspired exploration bonus that decays as f_best approaches the minimum.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP posterior mean at 2000 candidate points
    sigma : ndarray, shape (2000,)
        GP posterior standard deviation at 2000 candidate points
    f_best : float
        Best observed function value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean of coordinates scaled to [0,1])
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more desirable points
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        return np.zeros_like(mu)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # EI per unit cost
    ei_cost = ei / np.maximum(cost, 1e-9)
    
    # PSI-inspired exploration bonus:
    # The exploration signal should be strong when f_best is far from 0 (the minimum)
    # and decay as f_best approaches 0. This prevents over-exploitation near the minimum.
    # Use a scaled version of sigma that decays with f_best.
    
    # Normalize f_best to get a decay factor
    # When f_best is large (far from minimum), exploration bonus is strong
    # When f_best is small (near minimum), exploration bonus is weak
    # Use a soft decay: decay = 1 / (1 + f_best)
    decay = 1.0 / (1.0 + abs(f_best))
    
    # Exploration bonus based on uncertainty, scaled by decay
    # Use log(1+sigma) to get a bounded exploration signal
    exploration = decay * np.log1p(sigma_safe)
    
    # Normalize exploration to be on a similar scale as ei_cost
    # Find the max exploration to normalize
    max_expl = np.max(exploration)
    if max_expl > 1e-9:
        exploration_normalized = exploration / max_expl
    else:
        exploration_normalized = exploration
    
    # Blend EI/cost and exploration
    # Use a weight that favors exploration early and exploitation later
    # When f_best is large, weight exploration more heavily
    # When f_best is small, weight EI/cost more heavily
    w_explore = decay  # Between 0 and 1
    w_exploit = 1.0 - w_explore
    
    # Combine
    scores = w_exploit * ei_cost + w_explore * exploration_normalized
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
