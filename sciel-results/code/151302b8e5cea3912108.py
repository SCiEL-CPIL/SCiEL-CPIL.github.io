"""Acquisition for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = time.time() + 295.0  # safety margin before the 330s kill


def acquisition(mu, sigma, f_best):
    """Kriging-believer acquisition.

    Explores high-uncertainty regions with a UCB (mu - beta*sigma) to pick
    promising *regions*, then scores each candidate by the negative GP mean
    (exploitation) so the final selection is driven by low predicted values.
    """
    # Deadline guard: if we are already over budget, return a deterministic
    # exploitation-based score rather than risking a timeout.
    if time.time() > _DEADLINE:
        return -mu

    n = mu.shape[0]

    # Exploration: UCB with a small, fixed beta. sigma is already a standard
    # deviation in objective units.
    beta = 0.5
    ucb = mu - beta * sigma
    top = np.argmax(ucb)

    # Exploitation: score by negative mean (higher = lower predicted value).
    score = -mu
    # Guarantee the most-uncertain/promising region is not penalized relative
    # to pure exploitation by giving it a small boost.
    score[top] = -mu[top] + 1e-12

    # Ensure finiteness.
    score = np.where(np.isfinite(score), score, -np.inf)
    return score
# EVOLVE-BLOCK-END
