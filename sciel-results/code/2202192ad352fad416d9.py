"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Combines Expected Improvement (EI) with a UCB exploration term,
    using adaptive weighting based on uncertainty and progress.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions at candidate points.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    start_time = time.time()
    deadline = start_time + 270  # Safety margin well before 300s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Handle edge cases
    n = len(mu)
    if n == 0:
        return np.array([])

    # Replace near-zero sigma with small value to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # Compute Expected Improvement (EI) for exploitation
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei_raw = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei_raw, 0.0)

    # Compute UCB score for exploration: beta * sigma - mu
    beta = 1.5
    ucb_score = beta * sigma_safe - mu

    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei)
    if not np.isfinite(ei_max) or ei_max <= 1e-10:
        ei_max = 1.0
    ucb_max = np.max(ucb_score)
    if not np.isfinite(ucb_max) or ucb_max <= 1e-10:
        ucb_max = 1.0

    ei_norm = ei / ei_max
    ucb_norm = ucb_score / ucb_max

    # Adaptive blend: shift weight toward exploration when overall uncertainty is high
    mean_sigma = float(np.mean(sigma_safe))
    scale = max(abs(f_best), 1e-6)
    uncertainty_ratio = mean_sigma / (scale + mean_sigma + 1e-8)
    # ei_weight in [0.3, 0.6]; more uncertainty -> lower ei_weight (more exploration)
    ei_weight = 0.6 - 0.3 * uncertainty_ratio
    ucb_weight = 1.0 - ei_weight

    combined = ei_weight * ei_norm + ucb_weight * ucb_norm

    # Cost-aware: divide by cost
    scores = combined / cost_safe

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return scores

    return scores
# EVOLVE-BLOCK-END
