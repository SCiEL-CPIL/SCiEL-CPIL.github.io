"""Acquisition function for Bayesian optimization of sphere 1d using EHypR-inspired scoring."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 290s safety margin before 300s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Uses a structural combination of Expected Hypervolume Ratio (EHypR) logic
    with adaptive UCB-like exploration bonuses.
    
    The core idea: directly optimize for the expected improvement ratio,
    which naturally balances exploration and exploitation by weighting
    the probability of improvement by the magnitude of potential gain
    relative to the current best.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a safe fallback: just minimize mu (pure exploitation)
        return -mu.copy()
    
    # Clip sigma to avoid division by zero and ensure stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Normalize f_best to estimate progress
    # f_best starts around 25 (5^2) and approaches 0
    f_norm = np.clip(f_best / 25.0, 0, 1)
    
    # Standardize: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Compute CDF and PDF
    cdf_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # EHypR-inspired score:
    # We want to maximize E[max(f_best, f(x)) / f(x)] for minimization.
    # 
    # A practical approximation that captures the essence:
    # Score = P(f(x) < f_best) * (f_best - mu) / (f_best + epsilon)
    #       + P(f(x) < f_best) * sigma * phi(z) / (f_best + epsilon)
    #       + exploration_bonus * sigma * (1 - P(f(x) < f_best))
    #
    # The first two terms reward likely improvements scaled by their magnitude
    # relative to the current best. The third term provides exploration
    # in uncertain regions.
    
    eps = 1e-6
    f_safe = f_best + eps
    
    # Expected improvement component (probability-weighted gain)
    ei_component = cdf_z * (f_best - mu) / f_safe + cdf_z * sigma * phi_z / f_safe
    
    # Adaptive exploration bonus:
    # Higher when f_best is large (early), lower when f_best is small (late)
    # This naturally transitions from exploration to exploitation
    beta = 0.3 + 1.5 * (f_norm ** 1.0)
    
    # Exploration bonus: reward high sigma in regions unlikely to improve
    exploration_bonus = beta * sigma * (1 - cdf_z)
    
    # Combine: EHypR-inspired exploitation + adaptive exploration
    score = ei_component + exploration_bonus
    
    # Add a small penalty for extremely high sigma to prevent pure exploration
    # in far regions, keeping the search focused
    sigma_penalty = 0.02 * np.maximum(sigma - 1.0, 0) ** 2
    score = score - sigma_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
