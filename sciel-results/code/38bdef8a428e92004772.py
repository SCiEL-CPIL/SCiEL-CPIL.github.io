import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining soft multiplicative cost adjustment
    with a small exploration bonus.
    
    The multiplicative factor softly downweights expensive points without
    over-biasing toward cheap regions. A small sigma-based exploration term
    (also cost-adjusted) prevents premature convergence in early stages.
    """
    # Safety: ensure finite inputs and avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Compute z-score for Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # Compute Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Soft multiplicative cost adjustment:
    # cost is in [1, 2], so (cost - 1) is in [0, 1]
    # factor ranges from 1.0 (cost=1) to ~0.909 (cost=2)
    # This mildly penalizes expensive points without over-biasing
    alpha = 0.1
    cost_factor = 1.0 / (1.0 + alpha * (cost_safe - 1.0))
    
    # Apply multiplicative factor to EI
    ei_adjusted = ei * cost_factor
    
    # Small exploration bonus proportional to sigma, also cost-adjusted
    # This helps in early stages when EI is low
    exploration_weight = 0.03
    exploration = exploration_weight * sigma_safe * cost_factor
    
    # Combine adjusted EI with exploration bonus
    scores = ei_adjusted + exploration
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
