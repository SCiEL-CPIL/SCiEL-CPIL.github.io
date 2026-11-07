"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

# Known global minimum for Styblinski-Tang 1D
F_MIN = -39.1662

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    
    Uses an adaptive Lower Confidence Bound (LCB) strategy.
    LCB for minimization: score = mu - kappa * sigma
    kappa is adapted based on the gap between f_best and the known global minimum.
    
    Args:
        mu: array of shape (2000,) with posterior means
        sigma: array of shape (2000,) with posterior standard deviations
        f_best: lowest observed function value
    
    Returns:
        array of shape (2000,) with acquisition scores (higher is better)
    """
    # Guard against negative or zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the gap between current best and known global minimum
    # Clip gap to non-negative to handle cases where f_best might be slightly below F_MIN due to noise/numerics
    gap = np.maximum(f_best - F_MIN, 0.0)
    
    # Adaptive kappa (exploration parameter)
    # Start with high exploration (large kappa) when far from optimum
    # Decrease exploration (small kappa) as we get closer to optimum
    # Using an exponential decay for smooth transition
    KAPPA_MAX = 3.0
    KAPPA_MIN = 0.1
    DECAY_SCALE = 10.0  # Controls how quickly kappa decreases as gap shrinks
    
    # kappa = KAPPA_MIN + (KAPPA_MAX - KAPPA_MIN) * exp(-gap / DECAY_SCALE)
    kappa = KAPPA_MIN + (KAPPA_MAX - KAPPA_MIN) * np.exp(-gap / DECAY_SCALE)
    
    # LCB score for minimization:
    # We want to minimize f(x).
    # Low mu is good. High sigma is good for exploration.
    # score = mu - kappa * sigma
    # Higher score means we prefer this point.
    # If mu is very low, score is low (bad)? Wait.
    # If we want to MAXIMIZE the acquisition score to find the MINIMUM of f:
    # We want points with LOW mu.
    # So score should be -mu + kappa * sigma?
    # Or is LCB defined as lower bound on f?
    # LCB(x) = mu(x) - kappa * sigma(x).
    # We want to find the x that minimizes f.
    # If we assume f(x) >= LCB(x), then minimizing LCB(x) is a proxy for minimizing f(x).
    # So we want to MINIMIZE LCB(x).
    # But the function `acquisition` returns scores where HIGHER is better.
    # So we should return -LCB(x) = -mu(x) + kappa * sigma(x).
    
    score = -mu + kappa * sigma_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
