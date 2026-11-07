"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Global deadline guard: keep a safety margin before the 300 s kill.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # 20 s margin


def _time_guard():
    """Return True if we are within the safe time budget."""
    return time.time() < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using GP-UCB with budget-scaled beta.
    Returns 2000 finite scores; higher = evaluate next.
    """
    # Deadline guard: if running out of time, return a simple valid fallback.
    if not _time_guard():
        # Fallback: favor cheap, uncertain points
        c = np.maximum(cost, 1e-9)
        s = np.maximum(sigma, 1e-9)
        return s / c

    eps = 1e-12
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)

    # Estimate progress: f_best starts high (worst case) and decreases toward 0 (minimum)
    # For three-hump camel, f_best is typically in [0, ~1] after initial points
    # Use f_best to estimate how much of the budget has been "used" in terms of progress
    # When f_best is high (early), we want more exploration (higher beta)
    # When f_best is low (late), we want more exploitation (lower beta)
    
    # Normalize f_best to get a progress estimate
    # f_best=0 means we found the minimum (full exploitation)
    # f_best~1 or higher means we're still early (more exploration)
    # Use a smooth decay: beta goes from 3.5 (early) to 0.5 (late)
    f_best_clipped = np.clip(f_best, 0.0, 2.0)
    progress = 1.0 - f_best_clipped / 2.0  # 0 when f_best=2, 1 when f_best=0
    progress = np.clip(progress, 0.0, 1.0)
    
    # Linearly interpolate beta from 3.5 (early, high exploration) to 0.5 (late, exploitation)
    beta_high = 3.5
    beta_low = 0.5
    beta = beta_high + (beta_low - beta_high) * progress
    
    # GP-UCB for minimization: we want low mu + high sigma
    # UCB for minimization = mu - beta * sigma (lower is better, so we score higher when this is lower)
    # But we want higher score = more likely to evaluate
    # So: score = -(mu - beta * sigma) = -mu + beta * sigma
    # This means we prefer points with low mu (exploitation) and high sigma (exploration)
    
    ucb_score = -mu + beta * s
    
    # Cost-aware: divide by a convex combination of cost and a constant
    # This balances exploitation of low-cost regions against exploration
    # Using alpha * cost + (1-alpha) * 1.0
    # When alpha is high, cost matters more; when low, cost is less important
    alpha = 0.7
    cost_denom = alpha * c + (1.0 - alpha) * 1.0
    
    score = ucb_score / cost_denom
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
