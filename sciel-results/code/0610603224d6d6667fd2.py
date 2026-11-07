"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Uses a modified Expected Improvement (EI) that balances:
    1. Expected improvement over f_best
    2. Cost efficiency (lower cost is better)
    
    The cost adjustment uses a multiplicative factor rather than division to
    avoid over-biasing toward cheap regions while still favoring cost-efficiency.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure finite values
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Standard EI computation
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip EI to non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware adjustment:
    # Instead of dividing by cost (which can over-penalize expensive candidates),
    # use a softer multiplicative penalty.
    # The idea: scale EI by a factor that decreases with cost but not too aggressively.
    # cost is in [1, 2] since it's 1 + mean(x) where x in [0,1]^3
    # Normalize cost to [0, 1] range for the penalty
    cost_norm = (cost - 1.0) / 1.0  # now in [0, 1]
    
    # Multiplicative factor: starts at 1.0 for cost=1, decreases to ~0.5 for cost=2
    # This is softer than division by cost
    cost_factor = 1.0 - 0.3 * cost_norm  # ranges from 1.0 to 0.7
    
    # Also add a small exploration bonus for regions with high uncertainty
    # relative to the improvement, to encourage exploration
    # This helps when EI is small but sigma is large
    uncertainty_bonus = 0.1 * sigma / (np.abs(f_best - mu) + sigma + 1e-9)
    
    # Combine: base EI with cost adjustment plus a small uncertainty bonus
    score = ei * cost_factor + uncertainty_bonus * 1e-3
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
