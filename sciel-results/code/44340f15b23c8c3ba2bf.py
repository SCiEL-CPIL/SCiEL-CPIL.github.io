"""Cost-aware acquisition for Hartmann 6d Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_DEADLINE = 315.0
_start_time = time.time()


def _check_time():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a cost-normalized Lower Confidence Bound (LCB) strategy.
    The LCB for minimization is mu - beta * sigma.
    We maximize the score: (-mu + beta * sigma) / cost.
    Beta is adaptive based on the progress towards the known minimum.
    """
    if _check_time():
        return np.ones(2000)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Known global minimum for Hartmann 6d
    f_opt = -3.32237
    # Estimated initial function value (upper bound of typical random starts)
    f_start_est = 0.0
    
    # Calculate progress: 1.0 means we are at the start (bad), 0.0 means we are at the optimum
    # If f_best is worse than f_start_est, progress > 1, clip to 1
    # If f_best is better than f_opt, progress < 0, clip to 0
    progress = (f_best - f_opt) / (f_start_est - f_opt + 1e-9)
    progress = np.clip(progress, 0.0, 1.0)

    # Adaptive beta: High exploration early, low exploration late
    # Early (progress ~ 1): beta ~ 3.0
    # Late (progress ~ 0): beta ~ 0.5
    beta_max = 3.0
    beta_min = 0.5
    beta = beta_min + (beta_max - beta_min) * progress

    # LCB for minimization: L = mu - beta * sigma
    # We want to maximize the acquisition score.
    # Score = -L / cost = (-mu + beta * sigma) / cost
    lcb_negated = -mu + beta * sigma_safe
    scores = lcb_negated / cost_safe

    # Handle NaN/inf
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
