"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses a pure Lower Confidence Bound (LCB) strategy with a fixed, high
    exploration constant to aggressively target regions with high posterior variance.
    """
    t0 = time.time()
    DEADLINE = 295.0  # Safety margin before 300s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Ensure sigma is safe (positive)
    sigma_safe = np.maximum(sigma, 1e-8)

    # Fixed high exploration constant for aggressive uncertainty targeting
    k = 3.0

    # Lower Confidence Bound for minimization:
    # We want to minimize mu - k * sigma
    # So the score (higher is better) is -(mu - k * sigma) = -mu + k * sigma
    score = -mu + k * sigma_safe

    # Deadline guard
    if time.time() - t0 > DEADLINE:
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        return score

    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    return score
# EVOLVE-BLOCK-END
