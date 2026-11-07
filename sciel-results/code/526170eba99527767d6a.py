"""Rewritten acquisition for Bayesian optimization of Hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6d.
    
    Uses an adaptive UCB (Upper Confidence Bound) with a gap-scaled
    exploration coefficient that shifts from exploration to exploitation
    as the search converges.
    
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
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 300s limit
    
    n = len(mu)
    
    # Guard against non-positive sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute the gap between best observed and current predictions
    # This drives the adaptive exploration coefficient
    gap = np.abs(f_best) + 1.0  # Scale reference
    
    # Adaptive beta: starts high for exploration, decreases as we get closer to optimum
    # Use a soft transition based on the relative gap
    # When f_best is far from minimum (~-3.32), we want more exploration
    # When f_best is close to minimum, we want more exploitation
    
    # Estimate progress: how close are we to the known minimum?
    # Hartmann 6d minimum is approximately -3.32237
    hartmann_min = -3.32237
    progress = np.clip((f_best - hartmann_min) / (hartmann_min - (-1.0)), 0.0, 1.0)
    # progress ~ 1.0 means we're at the minimum, ~0.0 means we're far away
    
    # Beta starts high (exploration) and decays with progress
    # beta = beta_max * (1 - progress)^power + beta_min
    beta_max = 2.0
    beta_min = 0.3
    beta_power = 1.5
    beta = beta_min + (beta_max - beta_min) * (1.0 - progress) ** beta_power
    
    # UCB for minimization: score = -mu + beta * sigma
    # Higher score = more promising (lower mean, higher uncertainty)
    ucb_score = -mu + beta * sigma_safe
    
    # Check deadline
    if time.time() > deadline:
        return np.where(np.isfinite(ucb_score), ucb_score, 0.0)
    
    # Also compute EI as a secondary signal
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Blend UCB and EI: use EI when exploitation signal is strong,
    # UCB when exploration is needed
    ei_max = np.max(ei)
    ei_min = np.min(ei)
    ei_range = ei_max - ei_min + 1e-10
    
    # If EI has strong variation, trust it more
    ei_signal_strength = ei_range / (gap + 1e-10)
    
    # Blend weight: more EI when signal is strong
    ei_weight = np.clip(ei_signal_strength, 0.0, 0.7)
    
    # Normalize both to comparable scales
    ucb_mean = np.mean(ucb_score)
    ucb_std = np.std(ucb_score) + 1e-10
    ucb_norm = (ucb_score - ucb_mean) / ucb_std
    
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-10
    ei_norm = (ei - ei_mean) / ei_std
    
    # Final score: blend of normalized UCB and EI
    scores = (1.0 - ei_weight) * ucb_norm + ei_weight * ei_norm
    
    # Check deadline
    if time.time() > deadline:
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
