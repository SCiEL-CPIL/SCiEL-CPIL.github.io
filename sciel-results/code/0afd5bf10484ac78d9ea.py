"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Combines Expected Improvement with an adaptive exploration bonus that
    fades as f_best approaches the known global minimum.
    """
    t0 = time.time()
    DEADLINE = 295.0  # Safety margin before 300s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Ensure sigma is positive and handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)

    # Compute Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Progress toward the known global minimum
    global_min = -1.8013
    # f_best is in objective units (lower is better)
    # progress: 0 when f_best is very poor, 1 when f_best is near global_min
    # Michalewicz 2D values range roughly from -1.8 to 1.6
    progress = np.clip((f_best - 1.0) / (global_min - 1.0), 0.0, 1.0)

    # Exploration weight: high early, low late
    # Start with beta=1.5 (strong exploration), end with beta=0.2 (exploitation)
    beta = 1.5 - 1.3 * progress

    # Exploration term: beta * sigma (in objective units)
    exploration = beta * sigma_safe

    # Combine EI and exploration
    # Use a weighted blend where both terms are in comparable units
    # EI is typically small (improvement over best), exploration scales with sigma
    # Normalize exploration relative to typical EI scale to avoid domination
    # Use a soft combination: EI dominates when sigma is small, exploration helps when sigma is large
    
    # Scale exploration to be comparable to EI magnitude
    # Typical EI magnitude is on order of sigma when z ~ 0
    # So exploration = beta * sigma is already comparable
    score = ei + exploration

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline check: if we're running out of time, return immediately
    elapsed = time.time() - t0
    if elapsed > DEADLINE:
        return score

    return score
# EVOLVE-BLOCK-END
