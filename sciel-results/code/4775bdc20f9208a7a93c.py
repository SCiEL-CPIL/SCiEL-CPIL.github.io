"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Adaptive cost-aware acquisition for minimization.
    
    Combines:
    1. Cost-effective Expected Improvement (CEI)
    2. Adaptive LCB component that balances exploration/exploitation
    3. Penalty for high-cost regions
    
    Returns 2000 finite scores (higher = more promising to evaluate next).
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure no division by zero or negative costs
    cost_safe = np.maximum(cost, 1e-6)
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    diff = f_best - mu
    z = diff / sigma_safe
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    ei = diff * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI should be non-negative; clip to avoid tiny negatives from numerical error
    ei = np.maximum(ei, 0.0)
    
    # Cost-effective EI: EI per unit cost
    cei = ei / cost_safe
    
    # Adaptive LCB component:
    # LCB = mu - beta * sigma
    # beta is scaled by the uncertainty relative to the best observed improvement
    max_sigma = np.max(sigma_safe)
    # Scale beta: larger beta when uncertainty is high relative to potential improvement
    # Use a dynamic beta that adapts to the current state
    # beta = 1.0 * (max_sigma / (max_sigma + 1e-6)) gives ~1.0 when sigma is large
    # But we want to balance: use beta that decreases as we get closer to the optimum
    # Simple approach: beta = 0.5 to 1.0 range, scaled by uncertainty
    beta = 0.8 * (max_sigma / (max_sigma + abs(f_best) * 0.1 + 1e-6))
    beta = np.clip(beta, 0.3, 1.5)
    
    lcb = mu - beta * sigma_safe
    
    # Normalize LCB to be comparable to CEI
    # CEI has units of objective/cost, LCB has units of objective
    # Scale LCB by 1/cost to make them comparable
    lcb_normalized = -lcb / cost_safe  # Negate because lower LCB is better
    
    # Combine: weighted sum of CEI and normalized LCB
    # The weight can adapt: use more LCB when we're uncertain, more CEI when we're confident
    # Simple fixed weight: 0.7 * CEI + 0.3 * LCB_normalized
    # Or adaptive: weight based on the ratio of mean EI to mean LCB improvement
    score = 0.7 * cei + 0.3 * lcb_normalized
    
    # Additional penalty for very high cost regions
    # Penalize if cost is much higher than average
    mean_cost = np.mean(cost_safe)
    cost_penalty = 1.0 / (1.0 + 0.1 * np.maximum(0, cost_safe - mean_cost))
    score *= cost_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
