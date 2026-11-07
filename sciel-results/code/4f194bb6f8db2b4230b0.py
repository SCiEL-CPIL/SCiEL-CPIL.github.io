"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds of safety margin

# Cache for adaptive parameters
_adaptive_state = {"call_count": 0}


def _elapsed_time():
    return time.time() - _start_time


def _time_exceeded():
    return _elapsed_time() > (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-adjusted acquisition function for Bayesian optimization.
    
    Uses a hybrid approach:
    - Early phase: Strong exploration via UCB-like bonus
    - Late phase: Exploitation via Expected Improvement
    - Cost normalization throughout
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units
    sigma : array-like, shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Best (lowest) observed objective value
    cost : array-like, shape (2000,)
        Evaluation cost for each candidate (1 + mean(x_scaled))
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    _adaptive_state["call_count"] += 1
    
    # Convert to arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Check deadline
    if _time_exceeded():
        # Fallback: return simple cost-normalized negative mu
        return -mu / cost
    
    # Determine phase based on call count and f_best
    # Early phase: high uncertainty, f_best likely far from optimum
    # Late phase: low uncertainty, f_best close to optimum
    
    # Estimate progress: use f_best as a proxy
    # Ackley 2D minimum is 0, so f_best < 1 indicates progress
    progress = min(1.0, max(0.0, 1.0 - f_best / 10.0))
    
    # Blend between UCB (exploration) and EI (exploitation)
    # beta controls exploration-exploitation tradeoff
    # Decrease beta as we progress (less exploration needed)
    beta = 2.0 * (1.0 - 0.7 * progress)  # beta in [0.6, 2.0]
    
    # Compute UCB-like score: -mu + beta * sigma
    ucb_score = -mu + beta * sigma
    
    # Compute EI score
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Blend scores: more UCB early, more EI late
    blend = 0.5 * (1.0 - progress)  # blend in [0, 0.5]
    
    # Normalize EI to comparable scale
    # EI is in objective units, UCB is also in objective units
    # But EI can be negative or zero, so shift it
    ei_shifted = ei + np.abs(mu).mean() * 0.1  # small shift to make positive
    
    # Combine: weight UCB more in early phase
    combined = (1.0 - blend) * ucb_score + blend * ei_shifted
    
    # Cost normalization: divide by cost to account for heterogeneous costs
    # Use a power less than 1 to soften the cost penalty
    scores = combined / (cost ** 0.5)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Add a small exploration bonus for candidates with high sigma
    # This helps escape local optima
    sigma_bonus = 0.1 * sigma / cost
    scores = scores + sigma_bonus
    
    # Final safety: ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
