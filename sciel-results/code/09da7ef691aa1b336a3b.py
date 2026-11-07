"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track start time at module load
_T0 = time.time()
_DEADLINE = 300.0  # seconds


def _remaining():
    """Return seconds remaining before deadline."""
    return _DEADLINE - (time.time() - _T0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition score for 2000 candidates using Thompson Sampling.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed objective value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = prefer to evaluate next).
    """
    # Safety: check remaining time
    if _remaining() < 1.0:
        # Return something valid: uniform random scores
        return np.random.rand(2000)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Thompson Sampling: sample a single scalar offset from Gaussian
    # centered at f_best with std proportional to max uncertainty
    max_sigma = np.max(sigma_safe)
    # Scale the offset std to balance exploration/exploitation
    offset_std = max_sigma * 0.5
    offset = f_best + offset_std * np.random.randn()

    # Score: -(mu - offset) / cost
    # Higher score = more promising (lower mu relative to offset, lower cost)
    scores = -(mu - offset) / cost_safe

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
