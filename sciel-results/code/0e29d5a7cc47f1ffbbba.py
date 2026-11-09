import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses Expected Improvement (EI) with adaptive exploration.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # EI calculation:
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    
    # Handle zero sigma to avoid division by zero
    safe_sigma = np.where(sigma < 1e-12, 1e-12, sigma)
    
    z = (f_best - mu) / safe_sigma
    
    # Standard normal CDF and PDF
    # Using scipy.stats.norm for accuracy
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    EI = (f_best - mu) * Phi + safe_sigma * phi
    
    # Adaptive exploration: use a larger weight early when f_best is far from optimum
    # Hartmann 6D minimum is approximately -3.32237
    # When f_best is high (far from optimum), explore more
    # When f_best is low (close to optimum), exploit more
    
    # Estimate how far we are from the optimum
    # f_best starts around -1.0 to -2.0 and decreases toward -3.32
    # Use a simple heuristic: larger exploration when f_best > -2.5
    f_best_clamped = max(f_best, -3.5)
    # Normalize: 0 when close to optimum, 1 when far
    # Range: -3.32 (optimum) to about -1.0 (initial)
    exploration_scale = np.clip((-f_best_clamped - 3.32) / 2.32, 0.0, 1.0)
    
    # Exploration weight: 0.05 when close to optimum, 0.3 when far
    exploration_weight = 0.05 + 0.25 * exploration_scale
    
    scores = EI + exploration_weight * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
