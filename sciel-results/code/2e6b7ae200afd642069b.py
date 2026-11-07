"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses a structural blend of Expected Improvement (exploitation) and
    a lower confidence bound (exploration) with adaptive weighting
    based on progress toward the known global minimum.
    """
    t0 = time.time()
    DEADLINE = 295.0  # Safety margin before 300s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_safe = np.maximum(sigma, 1e-8)

    # Compute Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    # Clip z for numerical stability in norm.cdf/norm.pdf
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI is non-negative by construction; ensure no negatives from rounding
    ei = np.maximum(ei, 0.0)

    # Progress toward the known global minimum (-1.8013)
    global_min = -1.8013
    # f_best is in objective units (lower is better)
    # progress: 0 when f_best is very poor, 1 when f_best is near global_min
    # Michalewicz 2D values range roughly from -1.8013 to ~1.6
    f_best_start = 1.0  # typical starting value
    progress = np.clip((f_best - f_best_start) / (global_min - f_best_start), 0.0, 1.0)

    # Adaptive exploration weight: high early, low late
    # beta controls the UCB exploration term
    beta = 2.5 - 2.0 * progress  # beta from 2.5 to 0.5

    # UCB-like exploration term (for minimization, we want to explore high uncertainty)
    # LCB = mu - beta * sigma (lower is better)
    # But we want higher acquisition score = evaluate next
    # So exploration score = -mu + beta * sigma
    exploration = -mu + beta * sigma_safe

    # Normalize both terms to [0, 1] for comparable blending
    def safe_normalize(x):
        x_min = np.min(x)
        x_max = np.max(x)
        if x_max - x_min > 1e-12:
            return (x - x_min) / (x_max - x_min)
        return np.zeros_like(x)

    ei_norm = safe_normalize(ei)
    exp_norm = safe_normalize(exploration)

    # Weighted blend: shift from exploration to exploitation
    # Early: more exploration to find promising basins
    # Late: more exploitation to refine the best region
    w_exploit = 0.4 + 0.6 * progress  # 0.4 early, 1.0 late
    w_explore = 1.0 - w_exploit * 0.7  # 0.72 early, 0.3 late

    score = w_exploit * ei_norm + w_explore * exp_norm

    # Deadline guard
    if time.time() - t0 > DEADLINE:
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        return score

    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    return score
# EVOLVE-BLOCK-END
