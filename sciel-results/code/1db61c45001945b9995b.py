# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior
    f_best : float
        Best observed function value
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 250  # Safety margin before 300s limit
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Add small epsilon to avoid division by zero
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Clip f_best to a reasonable range
    f_best_clipped = np.clip(f_best, 0.0, 200.0)
    
    # Compute z-score for EI
    z = (f_best_clipped - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    ei = (f_best_clipped - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    scores_ei = ei / cost_safe
    
    # Compute a cost-aware UCB that balances exploration and exploitation
    # For minimization: we want low predicted value and high uncertainty
    # UCB for minimization: -mu + beta * sigma
    # beta adapts based on f_best: larger when f_best is large (early), smaller when f_best is small (late)
    
    # Smooth transition: beta starts high, decreases as f_best improves
    # Use a sigmoid-like transition centered around f_best ~ 5
    beta = 2.0 + 3.0 * np.exp(-f_best_clipped / 3.0)
    
    # UCB score for minimization (higher is better for acquisition)
    ucb = -mu + beta * sigma_safe
    
    # Cost-aware UCB
    scores_ucb = ucb / cost_safe
    
    # Compute a pure uncertainty term (for escaping local minima in rastrigin)
    scores_unc = sigma_safe / cost_safe
    
    # Adaptive blending:
    # Early (high f_best): more UCB and uncertainty for exploration
    # Late (low f_best): more EI for exploitation
    
    # Weight schedule based on f_best
    # w_ucb: high when f_best is large, low when f_best is small
    w_ucb = 0.4 + 0.4 * np.exp(-f_best_clipped / 2.0)
    w_ei = 1.0 - w_ucb - 0.1  # Reserve 10% for uncertainty
    w_unc = 0.1
    
    # Ensure weights are positive
    w_ucb = np.maximum(w_ucb, 0.05)
    w_ei = np.maximum(w_ei, 0.05)
    w_unc = np.maximum(w_unc, 0.05)
    
    # Normalize weights
    w_sum = w_ucb + w_ei + w_unc
    w_ucb /= w_sum
    w_ei /= w_sum
    w_unc /= w_sum
    
    # Combine scores
    scores = w_ei * scores_ei + w_ucb * scores_ucb + w_unc * scores_unc
    
    # Check time
    if time.time() > deadline:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Add a small penalty for very high cost to prefer cheaper evaluations
    # This helps with the cost-aware aspect more directly
    cost_penalty = 0.01 / cost_safe
    scores = scores - cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Clip to reasonable range to avoid numerical issues
    scores = np.clip(scores, -1e6, 1e6)
    
    return scores
# EVOLVE-BLOCK-END
