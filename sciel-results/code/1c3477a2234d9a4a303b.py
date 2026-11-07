"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = time.time() + 325.0


def _is_time_exceeded():
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for minimization of Branin 2D.
    Uses a combination of Expected Improvement and a tuned exploration term.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions at candidate points.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed value so far (lowest).
    
    Returns
    -------
    scores : np.ndarray, shape (2000,)
        Finite scores; higher means evaluate next.
    """
    if _is_time_exceeded():
        return np.zeros_like(mu)

    sigma_safe = np.maximum(sigma, 1e-9)

    # Z-score
    z = (f_best - mu) / sigma_safe

    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Upper Confidence Bound component for exploration
    # Use a time-adaptive beta: start with more exploration, reduce over time
    # We don't know the step count, so use a fixed moderate beta
    beta = 0.5
    ucb = mu + beta * sigma_safe  # For minimization, lower is better, so we want low mu + exploration
    # Actually for minimization, UCB should be mu - beta*sigma (optimistic lower bound)
    ucb_min = mu - beta * sigma_safe

    # Combine EI and a normalized exploration signal
    # Normalize EI to [0, 1] roughly based on the spread
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_norm = ei / ei_max

    # Exploration bonus: favor points with high sigma but not too high mu
    # Use sigma normalized
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 0 else 1.0
    sigma_norm = sigma_safe / sigma_max

    # Penalize points where mu is much worse than f_best
    # mu is in objective units, f_best is the best so far
    # If mu >> f_best, less interesting
    mu_penalty = np.exp(-np.maximum(mu - f_best, 0) / (sigma_safe + 1e-6))

    # Score: weighted combination
    # EI is the primary signal, exploration is secondary
    scores = ei_norm + 0.15 * sigma_norm * mu_penalty

    # Ensure finite and non-negative
    scores = np.where(np.isfinite(scores) & (scores >= 0), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
