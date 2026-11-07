"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses Thompson Sampling with adaptive temperature based on progress.
    """
    t0 = time.time()
    DEADLINE = 295.0  # Safety margin before 300s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_safe = np.maximum(sigma, 1e-8)

    # Progress toward the known global minimum (-1.8013)
    global_min = -1.8013
    f_best_start = 1.0  # typical starting value
    progress = np.clip((f_best - f_best_start) / (global_min - f_best_start), 0.0, 1.0)

    # Adaptive temperature for Thompson Sampling
    # Early: high temperature (more exploration)
    # Late: low temperature (more exploitation)
    # temperature scales the uncertainty term
    temperature = 1.0 - 0.7 * progress  # from 1.0 to 0.3

    # Draw a single random sample from the posterior
    # f_s = mu + sigma * xi, where xi ~ N(0, 1)
    xi = np.random.standard_normal(len(mu))
    f_sample = mu + temperature * sigma_safe * xi

    # For minimization, we want to select the point with the lowest sampled value
    # Score = -f_sample (higher score means lower sampled value, i.e., more promising)
    score = -f_sample

    # Deadline guard
    if time.time() - t0 > DEADLINE:
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        return score

    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    return score
# EVOLVE-BLOCK-END
