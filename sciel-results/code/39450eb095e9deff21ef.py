"""Cost-aware acquisition for Hartmann 3D: CA-UCB with adaptive beta and EI blending."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 290.0  # 10s safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining CA-UCB and Expected Improvement.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate, in [1, 2].
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    # Time guard: if we are close to the limit, return a simple fallback
    if time.time() > _DEADLINE:
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        cost_safe = np.maximum(cost, 1e-8)
        scores = ei / cost_safe
        scores[~np.isfinite(scores)] = 0.0
        return scores

    # Convert to numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Adaptive beta for CA-UCB
    # Use median cost as proxy for budget consumption
    # cost in [1, 2]; median_cost near 1 means early, near 2 means late
    median_cost = float(np.median(cost))
    
    # beta: starts high (exploration) when budget is fresh, decreases as budget consumed
    # beta = base / (1 + median_cost / 2.0) gives beta ~ 0.67 at start, ~0.33 at end
    # Let's use a more aggressive schedule: beta = 1.5 * (2.0 - median_cost) + 0.1
    # When median_cost=1: beta = 1.5*1 + 0.1 = 1.6
    # When median_cost=2: beta = 1.5*0 + 0.1 = 0.1
    beta = 1.5 * (2.0 - median_cost) + 0.1
    beta = float(np.clip(beta, 0.1, 1.6))
    
    # CA-UCB for minimization: score = -mu + beta * sigma
    ucb = -mu + beta * sigma_safe
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Normalize both to comparable scales
    ucb_mean = float(np.mean(ucb))
    ucb_std = float(np.std(ucb)) + 1e-8
    ucb_norm = (ucb - ucb_mean) / ucb_std
    
    ei_mean = float(np.mean(ei))
    ei_std = float(np.std(ei)) + 1e-8
    ei_norm = (ei - ei_mean) / ei_std
    
    # Blend: weight shifts from UCB (exploration) to EI (exploitation) as cost increases
    # weight_ei increases with median_cost
    weight_ei = float(np.clip(0.3 + 0.6 * (median_cost - 1.0), 0.3, 0.9))
    weight_ucb = 1.0 - weight_ei
    
    combined = weight_ucb * ucb_norm + weight_ei * ei_norm
    
    # Cost-aware: divide by cost
    scores = combined / cost_safe
    
    # Ensure all outputs are finite
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
