"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Quantile-Based Cost-Discounted Lower Confidence Bound.
    Uses a dynamic quantile (5th percentile) of the posterior predictive
    and applies a sigmoidal cost discount to prune high-cost candidates.
    """
    start_time = time.time()
    deadline = start_time + 300.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    if mu.size == 0:
        return np.array([])

    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # Compute the 5th percentile of the posterior predictive distribution:
    # For N(mu, sigma^2), the q-th quantile is mu + sigma * z_q
    # where z_q is the q-th quantile of the standard normal.
    # 5th percentile -> z_q = norm.ppf(0.05) ≈ -1.6449
    z_quantile = norm.ppf(0.05)
    
    # The quantile-based lower bound (lower is better for minimization)
    # q_lcb = mu + sigma * z_quantile
    # Since z_quantile is negative, this gives a value below mu
    q_lcb = mu + sigma_safe * z_quantile

    # For a maximization score (higher = better), we negate:
    # score_raw = -q_lcb = -mu - sigma * z_quantile = -mu + sigma * |z_quantile|
    score_raw = -mu - sigma_safe * z_quantile

    # Sigmoidal cost discount: aggressively prune high-cost candidates
    # Cost ranges from 1 to 2 (1 + mean(x) where x in [0,1])
    # We want to strongly penalize costs near 2 and lightly penalize costs near 1
    # Use a sigmoid centered at cost=1.5 with steepness to create a sharp cutoff
    cost_center = 1.5
    cost_steepness = 10.0
    cost_discount = 1.0 / (1.0 + np.exp(cost_steepness * (cost_safe - cost_center)))
    
    # Apply the cost discount multiplicatively
    score = score_raw * cost_discount

    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline check
    if time.time() > deadline - 5.0:
        return score

    return score
# EVOLVE-BLOCK-END
