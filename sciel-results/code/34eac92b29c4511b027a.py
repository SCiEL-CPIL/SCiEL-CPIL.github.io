"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using quantile-based heuristic.
    
    Ranks candidates by the ratio of their posterior probability of improving
    over f_best to their evaluation cost. This decouples the search from noisy
    mean estimates and prioritizes low-cost regions with high improvement likelihood.
    """
    t_start = time.time()
    
    # Safety: ensure sigma is positive
    sigma_safe = np.clip(sigma, 1e-8, None)
    
    # Compute z-score: z = (f_best - mu) / sigma
    # P(f(x) < f_best) = Phi(z) where z = (f_best - mu) / sigma
    delta = f_best - mu
    z = delta / sigma_safe
    
    # Clip z for numerical stability
    z_clipped = np.clip(z, -10, 10)
    
    # Probability of improvement: P(f(x) < f_best) = Phi(z)
    p_improve = norm.cdf(z_clipped)
    
    # Ensure p_improve is in [0, 1]
    p_improve = np.clip(p_improve, 0.0, 1.0)
    
    # Cost normalization
    cost_safe = np.clip(cost, 1.0, None)
    
    # Primary score: probability of improvement per unit cost
    score = p_improve / cost_safe
    
    # Add a small exploration bonus based on uncertainty
    # This helps avoid getting stuck in local optima
    exploration_bonus = sigma_safe / (cost_safe * 50.0)
    
    score = score + exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() - t_start > 325:
        pass
    
    return score
# EVOLVE-BLOCK-END
