"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_t0 = time.time()
_DEADLINE = 300.0  # seconds
_MARGIN = 5.0      # safety margin in seconds


def _safe_z(f_best, mu, sigma):
    """Compute z = (f_best - mu) / sigma with safe division."""
    sigma_safe = np.maximum(sigma, 1e-9)
    return (f_best - mu) / sigma_safe


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware blended acquisition: EI/cost + bounded UCB exploration term.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard
    if time.time() - _t0 > _DEADLINE - _MARGIN:
        # Return a simple, fast fallback
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei / np.maximum(cost, 1e-9)

    n = len(mu)

    # Safe sigma
    sigma_safe = np.maximum(sigma, 1e-9)

    # Z-scores
    z = (f_best - mu) / sigma_safe

    # Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Cost-normalized EI
    cost_safe = np.maximum(cost, 1e-9)
    ei_cost = ei / cost_safe

    # UCB exploration term: beta * sigma / cost
    # Use a moderate beta that balances exploration
    # Scale beta by the typical improvement magnitude to keep it bounded
    beta = 0.5
    ucb = beta * sigma_safe / cost_safe

    # Blend: primarily EI/cost with a bounded exploration bonus
    scores = ei_cost + ucb

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
