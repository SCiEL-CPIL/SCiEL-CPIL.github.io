"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track start time for deadline guard
_start_time = time.time()
_DEADLINE = 330.0  # wall-clock limit
_SAFETY_MARGIN = 15.0  # seconds to leave before the limit

def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score for 2000 candidate points.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed value so far
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = better)
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Avoid division by zero in sigma and cost
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard EI for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI should be non-negative, but numerical errors might make it slightly negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    score_ei = ei / cost_safe
    
    # UCB component for exploration (Minimization version: Lower Confidence Bound)
    # We want to explore where the lower bound is low.
    # LCB = mu - beta * sigma
    # Potential improvement = f_best - LCB
    # We use beta = 1.5 based on insights that UCB(1.0-2.0) is effective
    beta = 1.5
    lcb = mu - beta * sigma_safe
    potential_improvement_ucb = f_best - lcb
    # Clip to 0 to avoid rewarding points that are already worse than f_best optimistically
    potential_improvement_ucb = np.maximum(potential_improvement_ucb, 0.0)
    
    # Cost-aware UCB
    score_ucb = potential_improvement_ucb / cost_safe
    
    # Blend EI and UCB
    # Equal weighting to balance exploitation (EI) and exploration (UCB)
    # No normalization to preserve relative magnitude of improvements
    weight_ei = 0.5
    weight_ucb = 0.5
    
    scores = weight_ei * score_ei + weight_ucb * score_ucb
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time check
    elapsed = time.time() - _start_time
    if elapsed > (_DEADLINE - _SAFETY_MARGIN):
        return scores
    
    return scores
# EVOLVE-BLOCK-END
