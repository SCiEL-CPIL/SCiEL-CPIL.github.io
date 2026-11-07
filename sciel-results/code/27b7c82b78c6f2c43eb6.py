"""Cost-aware acquisition for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 295.0  # safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a combination of Expected Improvement
    and cost-aware weighting for efficient Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions.
    sigma : array of shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more desirable).
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Return a safe fallback: prefer low mu and low cost
        mu_safe = np.asarray(mu, dtype=np.float64)
        cost_safe = np.asarray(cost, dtype=np.float64)
        mu_safe = np.where(np.isfinite(mu_safe), mu_safe, 0.0)
        cost_safe = np.where(np.isfinite(cost_safe) & (cost_safe > 0), cost_safe, 1.0)
        scores = -mu_safe / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-6, 1e2)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)  # avoid extreme values
    
    # Standard normal CDF and PDF
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    EI = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Cost-aware weighting: we want to prefer candidates that offer
    # high EI relative to their cost. Use a power-based cost penalty.
    # Normalize cost to [0, 1] range for consistent scaling
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range > 1e-10:
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.zeros_like(cost)
    
    # Cost penalty: higher cost gets lower weight
    # Use a power law: weight = (1 - cost_norm)^beta
    # beta controls how strongly we penalize expensive candidates
    beta = 1.5
    cost_weight = np.power(1.0 - cost_norm, beta)
    
    # Ensure cost_weight is finite and positive
    cost_weight = np.where(np.isfinite(cost_weight) & (cost_weight > 0), cost_weight, 1e-6)
    
    # Combine EI with cost weighting
    # Score = EI / cost_effective, where cost_effective incorporates the penalty
    # Using division by cost directly as a simple cost-awareness mechanism
    scores = EI / (cost * (1.0 + cost_norm * 0.5))
    
    # Add a small exploration bonus for high uncertainty regions
    # This helps escape local optima in early iterations
    uncertainty_bonus = sigma * 0.1
    scores += uncertainty_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
