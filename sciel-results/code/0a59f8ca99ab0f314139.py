"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a hybrid approach:
    - UCB term for exploration (helps find the shifted minimum)
    - Cost penalty to avoid expensive regions
    - Safety margin for time
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Evaluation cost for each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition values (higher = evaluate next)
    """
    start_time = time.time()
    deadline = start_time + 280.0  # 280s safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Handle edge cases
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for EI
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = improvement * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # UCB term: balance exploration and exploitation
    # Beta controls exploration; higher beta = more exploration
    beta = 2.0
    
    # UCB score
    ucb = mu + beta * sigma_safe
    
    # Cost-aware: penalize expensive evaluations
    # cost is in [1, 2] based on the problem description
    # Normalize cost to [0, 1] for the penalty
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + 1e-9
    cost_norm = (cost - cost_min) / cost_range
    
    # Cost penalty: higher cost = lower score
    # Use a multiplicative penalty to keep scores well-scaled
    cost_penalty = 1.0 / (1.0 + 0.5 * cost_norm)
    
    # Combine EI and UCB with cost awareness
    # Weight: 0.5 for exploitation (EI), 0.5 for exploration (UCB)
    # Scale EI to be comparable to UCB
    ei_scale = np.max(np.abs(mu)) + np.max(sigma_safe) + 1e-9
    ei_scaled = ei / ei_scale
    
    # Final score: blend of EI and UCB, adjusted for cost
    score = (0.6 * ei_scaled + 0.4 * (1.0 - ucb)) * cost_penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 270.0:
        # Return quickly if we're running out of time
        return score
    
    return score
# EVOLVE-BLOCK-END
