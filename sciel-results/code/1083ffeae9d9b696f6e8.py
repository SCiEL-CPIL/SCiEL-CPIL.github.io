"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: blends UCB (exploration) with EI (exploitation).
    Uses an adaptive beta that starts high (exploratory) and decays toward
    exploitation as f_best improves (i.e., as we get closer to the optimum).
    
    mu: array of shape (N,) - posterior mean
    sigma: array of shape (N,) - posterior std
    f_best: scalar - best observed value so far
    
    Returns: array of shape (N,) with higher scores preferred.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    # Safety: clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute EI component
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB component: mu + beta * sigma
    # Adaptive beta: start exploratory, decay toward exploitation
    # The "progress" is how much f_best has decreased from initial scale
    # Beale function minimum is 0, and typical values are O(1-10)
    # Use a smooth schedule based on f_best magnitude
    
    # When f_best is large (early), we want more exploration
    # When f_best is small (late), we want more exploitation
    # beta ranges from ~2.5 (very exploratory) to ~0.5 (exploitative)
    # Use a smooth decay: beta = 0.5 + 2.0 * exp(-f_best / scale)
    # For Beale, f_best starts around 5-20 and goes to ~0
    # scale ~ 3 gives good behavior
    
    scale = 3.0
    beta = 0.5 + 2.0 * np.exp(-max(f_best, 0.0) / scale)
    
    ucb = mu + beta * sigma_safe
    
    # Normalize both to comparable scales for blending
    # EI is in objective units (improvement), UCB is in objective units (value)
    # We want higher acquisition = better candidate
    # EI: higher is better (more expected improvement)
    # UCB: we want to pick high UCB for exploration, but for minimization,
    #       we actually want low predicted values. UCB for minimization:
    #       score = -mu + beta * sigma (we want low mu, high sigma)
    
    # Actually, for minimization:
    # UCB score = -mu + beta * sigma (higher = more promising for minimization)
    # EI score = expected improvement = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    ucb_min = -mu + beta * sigma_safe
    
    # Blend: weight EI more as we converge
    # w_ei increases as f_best decreases
    w_ei = 0.3 + 0.7 * (1.0 - np.exp(-max(f_best, 0.0) / 2.0))
    w_ucb = 1.0 - w_ei
    
    # Normalize to zero mean, unit variance for blending
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-12
    ucb_mean = np.mean(ucb_min)
    ucb_std = np.std(ucb_min) + 1e-12
    
    ei_norm = (ei - ei_mean) / ei_std
    ucb_norm = (ucb_min - ucb_mean) / ucb_std
    
    score = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
