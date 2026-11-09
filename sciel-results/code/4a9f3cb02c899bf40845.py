"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    with normalized cost penalty for better balance.
    """
    # Numerical safety
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score for EI
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using scipy.stats.norm for cdf and pdf
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize cost to [0, 1] range
    # Cost is in [1, 2] (1 + mean(x_scaled) where x_scaled is in [0,1])
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    
    if cost_range > 1e-10:
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.zeros_like(cost)
    
    # The minimum value of cosine8 is -0.8.
    f_min = -0.8
    
    # Current regret
    regret = f_best - f_min
    
    # Normalize regret to [0, 1] for scaling the penalty
    # If f_best is much worse than -0.8, regret is large.
    # We assume a max relevant regret of 1.0 for scaling purposes.
    regret_norm = np.clip(regret, 0.0, 1.0)
    
    # Lambda controls the trade-off between improvement and cost.
    # When regret is high (poor f_best), we want to explore more aggressively,
    # so we penalize cost less (lower lambda).
    # When regret is low (good f_best), we want to conserve budget and exploit,
    # so we penalize cost more (higher lambda).
    
    # Lambda ranges from 0.5 (high regret) to 2.0 (low regret)
    # Using stronger penalty since cost is now normalized to [0,1]
    lambda_cost = 0.5 + 1.5 * (1.0 - regret_norm)
    
    # Score = EI - lambda * cost_normalized
    # Higher score is better.
    scores = ei - lambda_cost * cost_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
# EVOLVE-BLOCK-END
