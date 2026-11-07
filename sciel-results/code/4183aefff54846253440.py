"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard state
_start_time = [None]
_deadline = 300.0  # 300 seconds wall-clock limit

def _get_elapsed():
    """Get elapsed time since the first call, initializing if needed."""
    if _start_time[0] is None:
        _start_time[0] = time.time()
    return time.time() - _start_time[0]

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are close to the limit, return a fast fallback
    elapsed = _get_elapsed()
    if elapsed > _deadline - 5.0:
        # Safe fallback: simple EI/cost computed quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB for minimization: score = mu - beta * sigma
    # We want to minimize, so lower mu and higher sigma (uncertainty) is good for exploration
    # Standard UCB for minimization: mu - beta * sigma
    # However, we want to maximize the acquisition score.
    # So we use -mu + beta * sigma.
    
    # Adaptive Beta: use the 95th percentile of the z-scores to set the exploration level
    # This makes the exploration scale with the current uncertainty relative to improvement
    z_abs = np.abs(z)
    beta = np.percentile(z_abs, 95) if np.max(z_abs) > 0 else 1.0
    # Cap beta to prevent excessive exploration in early stages if z is huge
    beta = min(beta, 5.0)
    
    ucb = -mu + beta * sigma_safe
    
    # Structural Change: Percentile Normalization
    # This makes the scores robust to outliers and ensures a consistent spread of scores
    # Rank the candidates by EI and UCB
    
    ei_rank = np.argsort(np.argsort(ei)) / (len(ei) - 1)
    ucb_rank = np.argsort(np.argsort(ucb)) / (len(ucb) - 1)
    
    # Combine using a weighted sum of ranks
    # This is equivalent to a robust linear combination
    weight_ei = 0.6
    weight_ucb = 0.4
    
    score = weight_ei * ei_rank + weight_ucb * ucb_rank
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # Use a stronger cost discount to encourage exploring cheap regions first
    # Cost is 1 + mean(x_scaled), so range is [1, 2]
    # (cost_safe ** 1.0) is linear penalty
    alpha = 1.0
    score = score / (cost_safe ** alpha)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
