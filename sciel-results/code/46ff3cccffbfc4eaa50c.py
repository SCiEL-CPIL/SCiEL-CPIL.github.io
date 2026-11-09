"""Cost-aware acquisition for Hartmann-3D BO loop."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = time.time() + 290.0  # safety margin before 300s wall-clock limit


def _quantile_lcb(mu, sigma, f_best, q=0.05):
    """
    Quantile-based LCB: map posterior to a low quantile value.
    For minimization, we want low values, so we use the q-th percentile
    of the predictive distribution N(mu, sigma^2).
    The q-th percentile is mu + sigma * norm.ppf(q).
    We negate it for a higher-is-better acquisition score.
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    z_q = norm.ppf(q)  # negative for q < 0.5
    # The q-th percentile of N(mu, sigma^2) is mu + sigma * z_q
    quantile_val = mu + sigma_safe * z_q
    # For minimization, lower quantile_val is better, so we negate
    return -quantile_val


def acquisition(mu, sigma, f_best, cost):
    """
    Quantile-Based Cost-Discounted Lower Confidence Bound.
    
    Uses a dynamic quantile (5th percentile) for the posterior predictive
    and applies a sigmoidal cost discount to aggressively prune high-cost
    candidates in outer regions of the domain.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of scores (higher = evaluate next)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Safe fallback: simple quantile LCB divided by cost
        sigma_safe = np.maximum(sigma, 1e-12)
        z_q = norm.ppf(0.05)
        quantile_val = mu + sigma_safe * z_q
        score = -quantile_val / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(score), score, 0.0)
    
    # Handle edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    sigma = np.maximum(sigma, 1e-12)
    
    # Quantile-based LCB: use 5th percentile of predictive distribution
    # This is more aggressive than standard LCB with linear sigma
    q = 0.05
    z_q = norm.ppf(q)  # approximately -1.645
    quantile_val = mu + sigma * z_q
    lcb_score = -quantile_val  # higher is better
    
    # Sigmoidal cost discount: aggressively penalize high-cost points
    # Cost ranges from 1 to 4 (since cost = 1 + mean(x) and x in [0,1]^3)
    # Normalize cost to [0, 1] range for sigmoid
    cost_norm = (cost - 1.0) / 3.0  # maps [1, 4] to [0, 1]
    cost_norm = np.clip(cost_norm, 0.0, 1.0)
    
    # Sigmoidal discount: sharp transition to penalize expensive points
    # At cost_norm=0 (cheapest), discount ~ 0.5 + 0.5 = 1.0 (full score)
    # At cost_norm=1 (most expensive), discount ~ 0.5 - 0.5 = 0.0 (pruned)
    # Use a steeper sigmoid for aggressive pruning
    steepness = 8.0
    discount = 0.5 * (1.0 + np.tanh(steepness * (0.5 - cost_norm)))
    
    # Apply the discount to the LCB score
    score = lcb_score * discount
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
