"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: 330s wall-clock. Each call should be fast (< 0.1s).
# We track a global start time to ensure we don't exceed the budget.
# However, since the function is called 75 times total (3 seeds * 25 steps),
# and each call is O(N) with N=2000, the total time is negligible.
# We include a safety check just in case.
_START_TIME = time.time()
_DEADLINE = 330.0 - 5.0  # 5s safety margin

def acquisition(mu, sigma, f_best):
    """
    Computes an acquisition score for each of the 2000 candidates.
    
    Parameters:
    mu: array of shape (2000,), posterior mean
    sigma: array of shape (2000,), posterior std
    f_best: float, best observed value so far
    
    Returns:
    scores: array of shape (2000,), higher is better
    """
    # Safety check: if we are close to the deadline, return a simple score
    if time.time() > _DEADLINE:
        return -mu.copy()

    # Ensure sigma is positive to avoid log(0) or division issues
    sigma = np.maximum(sigma, 1e-10)

    # Strategy: Quantile-based acquisition (Q0.1)
    # This targets the extreme lower tail of the posterior.
    # For a normal distribution N(mu, sigma^2), the q-quantile is:
    # x_q = mu - sigma * z_q
    # where z_q = norm.ppf(q) is negative for q < 0.5.
    # We want to minimize, so we prefer low values.
    # The "score" for "higher is better" is the negative of the quantile.
    # score = -(mu - sigma * z_q) = -mu + sigma * z_q
    # Since z_q is negative, this is -mu - sigma * |z_q|.
    # This is effectively an LCB with beta = |z_q|.
    
    # The insight from the cell findings suggests that Q0.1 (quantile 0.1)
    # is a strong structural choice.
    # z_0.1 = norm.ppf(0.1) ≈ -1.28155
    
    # Let's verify the logic.
    # If we pick the point with the lowest 10th percentile of the posterior,
    # we are betting that there is a 10% chance the true value is below that quantile.
    # This is more conservative than the mean (Q0.5) and less aggressive than the minimum (Q0.0).
    # For minimization, we want the lowest possible value.
    # The 10th percentile is a good balance between exploitation (low mu) and exploration (high sigma).
    
    z_q = norm.ppf(0.1)
    
    # Score = -(mu + sigma * z_q)
    # Since z_q is negative, let beta = -z_q > 0.
    # Score = -mu + sigma * beta
    beta = -z_q  # ≈ 1.28
    
    scores = -mu + beta * sigma
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores

# EVOLVE-BLOCK-END
