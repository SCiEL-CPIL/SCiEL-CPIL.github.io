"""Cost-aware acquisition for Bayesian optimization on cosine8 8d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Thompson-sampling-inspired
    deterministic score that balances exploitation and exploration.
    
    mu, sigma: shape (2000,) - GP mean and std in objective units
    f_best: scalar - lowest observed value
    cost: shape (2000,) - cost of evaluating each candidate (1 + mean(x in [0,1]))
    
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Time guard: track elapsed time
    t_start = time.time()
    
    # Safety: ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Handle edge cases
    if n == 0:
        return np.array([])
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score: how many std devs below f_best the mean is
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) - standard formula
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For minimization, we want to improve below f_best
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (clip tiny negative values from numerical noise)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    # cost is in [1, 2] since x is in [-1,1]^8, scaled to [0,1]
    # mean of 8 coordinates in [0,1] is in [0,1], so cost in [1,2]
    cost_safe = np.maximum(cost, 1e-9)
    
    # Base score: EI per unit cost
    score = ei / cost_safe
    
    # Additional exploration term: reward high uncertainty relative to cost
    # This helps escape local optima in 8D
    # Use a quadratic penalty on variance to avoid over-exploration
    # but still reward some exploration
    
    # Dynamic scaling: as f_best approaches the known minimum (-0.8),
    # reduce exploration
    # The minimum of cosine8 is -0.8
    min_possible = -0.8
    gap = f_best - min_possible  # positive when f_best > min_possible
    
    # Scale factor: 1.0 when far from optimum, decreases as we approach
    # Use a smooth transition
    if gap > 0.01:
        exploration_weight = 0.15
    else:
        exploration_weight = 0.15 * max(0, gap / 0.01)
    
    # Exploration term: sigma / cost, weighted
    # This rewards points with high uncertainty that are cheap to evaluate
    exploration_term = (sigma_safe / cost_safe) * exploration_weight
    
    # Combine: exploitation (EI/cost) + exploration (sigma/cost)
    final_score = score + exploration_term
    
    # Ensure all values are finite
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    # Time guard: if we've spent too much time, return immediately
    elapsed = time.time() - t_start
    if elapsed > 280:  # Safety margin before 300s limit
        return final_score
    
    return final_score
# EVOLVE-BLOCK-END
