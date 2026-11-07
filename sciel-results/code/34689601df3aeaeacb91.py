"""Combined acquisition function for Hartmann 6D optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: 330s wall-clock limit, use 325s to be safe
_DEADLINE = time.time() + 325.0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and a
    saturated exploration term.

    This approach balances exploitation (EI) with exploration (sigma)
    using a tanh saturation to prevent high-uncertainty regions from
    dominating the score, which is beneficial for multi-modal functions
    like Hartmann 6D.

    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    # Time check: if we're close to deadline, return a simple, fast score
    if time.time() > _DEADLINE - 5.0:
        mu = np.asarray(mu, dtype=np.float64)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        return -mu

    # Ensure inputs are numpy arrays of correct type
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)

    # 1. Compute Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # 2. Compute Saturated Exploration Term
    # Use tanh to bound the exploration bonus.
    # The scale factor (0.5) determines how quickly saturation occurs.
    # A smaller scale means the term saturates faster, limiting the influence
    # of very high sigma values.
    exploration_scale = 0.5
    exploration_weight = 0.1
    exploration_term = exploration_weight * np.tanh(sigma / exploration_scale)

    # 3. Combine
    scores = ei + exploration_term

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
