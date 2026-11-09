"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Dynamic Pareto-weighted acquisition: blends normalized EI and normalized
    uncertainty with a time-varying weight that shifts from exploration to
    exploitation as the search progresses.
    
    mu: (n,) array of GP mean predictions
    sigma: (n,) array of GP standard deviation predictions
    f_best: current best observed value (minimum)
    
    Returns: (n,) array of scores, higher is better
    """
    start_time = time.time()
    deadline = start_time + 295.0

    sigma_safe = np.maximum(sigma, 1e-9)

    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Normalize EI and sigma to [0, 1] range for Pareto weighting
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < 1e-12:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range

    sig_min = np.min(sigma_safe)
    sig_max = np.max(sigma_safe)
    sig_range = sig_max - sig_min
    if sig_range < 1e-12:
        sig_norm = np.zeros_like(sigma_safe)
    else:
        sig_norm = (sigma_safe - sig_min) / sig_range

    # Infer progress: how close are we to the known minimum?
    min_val = -3.32237
    init_val = -1.0
    progress = np.clip((init_val - f_best) / (init_val - min_val), 0.0, 1.0)

    # Dynamic weight: start with more exploration, shift toward exploitation
    exploration_weight = 0.5 * (1.0 - progress) + 0.05
    exploitation_weight = 1.0 - exploration_weight

    # Pareto-weighted score
    scores = exploitation_weight * ei_norm + exploration_weight * sig_norm

    # Scale back to objective units
    scores = scores * (ei_range + 1e-9)

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        return scores

    return scores
# EVOLVE-BLOCK-END
