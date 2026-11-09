"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: 330s wall-clock limit, use 320s safety margin
_DEADLINE = 320.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    Combines Expected Improvement (EI) with UCB exploration and cost penalty.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if mu.size == 0:
        return np.array([])
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # UCB term for exploration
    # Use a moderate beta that balances exploitation and exploration
    beta = 2.0
    ucb = mu + beta * sigma_safe
    
    # Since we're minimizing, lower ucb is better, so we negate for scoring
    # Actually, for minimization: we want to find points where mu is low
    # UCB for minimization: mu - beta * sigma (lower is better)
    # But we want higher score = better, so: -(mu - beta * sigma) = -mu + beta * sigma
    ucb_score = -mu + beta * sigma_safe
    
    # Normalize both EI and UCB to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_norm = ei / ei_max
    
    ucb_max = np.max(ucb_score) if np.max(ucb_score) > 0 else 1.0
    ucb_min = np.min(ucb_score)
    ucb_range = ucb_max - ucb_min if ucb_range > 0 else 1.0
    ucb_norm = (ucb_score - ucb_min) / ucb_range
    
    # Combine EI and UCB with weights
    # Weight EI more heavily since it's directly related to improvement
    w_ei = 0.6
    w_ucb = 0.4
    combined = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Cost-aware scoring: divide by cost to prefer cheaper evaluations
    # Cost is in [1, 2] range (1 + mean(x_scaled_to_01))
    # Using power 1.0 for balanced cost penalty
    cost_penalty = np.power(cost, 1.0)
    
    # Score = combined / cost_penalty
    scores = combined / cost_penalty
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
