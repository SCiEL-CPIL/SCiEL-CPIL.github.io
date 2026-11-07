"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: blends expected improvement with uncertainty
    to balance exploitation and exploration, normalized by cost.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Uncertainty bonus: encourages exploration in high-uncertainty regions
    # Scale uncertainty relative to the range of f_best
    # Use a tempered uncertainty term
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    # Normalize sigma to [0, 1] for the bonus
    sigma_norm = sigma_safe / (sigma_max + 1e-9)
    
    # Blend: primarily EI, with a smaller uncertainty component
    # The uncertainty term helps escape local optima
    alpha = 0.3  # weight for uncertainty
    score = ei + alpha * sigma_norm * sigma_mean
    
    # Cost-aware normalization
    score = score / cost_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
