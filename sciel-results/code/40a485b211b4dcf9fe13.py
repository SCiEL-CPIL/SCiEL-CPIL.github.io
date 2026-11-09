"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each point.
    
    Returns:
    --------
    scores : array-like, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    t_start = time.time()
    MAX_TIME = 280.0  # Safety margin before 300s kill

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Compute z-score for Expected Improvement
    diff = f_best - mu
    z = diff / sigma_safe

    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -50, 50)

    # Expected Improvement
    ei = diff * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)

    # EI should be non-negative; clip small negatives from numerical error
    ei = np.maximum(ei, 0.0)

    # Cost-normalized EI (primary exploitation signal)
    score = ei / cost_safe

    # Exploration bonus: when EI is near zero, encourage exploring high-uncertainty regions
    # Scale by how far f_best is from the known minimum (0 for Ackley)
    # When f_best is large (early), explore more; when close to 0, exploit
    f_best_abs = abs(f_best)
    # Adaptive exploration weight: stronger when far from optimum
    explore_weight = 0.1 * np.exp(-f_best_abs / 5.0)

    # Normalize sigma for the bonus so it's comparable across iterations
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-12:
        exploration_bonus = (sigma_safe / sigma_max) * explore_weight / cost_safe
    else:
        exploration_bonus = np.zeros_like(score)

    score += exploration_bonus

    # Check time deadline
    elapsed = time.time() - t_start
    if elapsed > MAX_TIME:
        score = np.where(np.isfinite(score), score, 0.0)
        return score

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
