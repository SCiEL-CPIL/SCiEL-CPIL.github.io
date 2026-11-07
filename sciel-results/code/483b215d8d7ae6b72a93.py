"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on Expected Improvement (EI)
    normalized by cost, with an adaptive exploration component.
    
    The strategy uses EI to balance exploration (via sigma) and exploitation
    (via f_best - mu). Dividing by cost accounts for the varying evaluation
    cost in the domain. We add a small UCB-like exploration term to avoid
    getting stuck in local exploitation, weighted by inverse cost.
    """
    # Ensure sigma is strictly positive to avoid division by zero in z
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard EI calculation
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware EI
    cost_safe = np.maximum(cost, 1e-9)
    ei_cost = ei / cost_safe
    
    # Add a small exploration bonus based on uncertainty, also cost-aware
    # This helps escape local exploitation traps
    # Use a damped sigma term to provide exploration without dominating
    exploration_bonus = 0.1 * sigma_safe / cost_safe
    
    # Combine exploitation (EI/cost) and exploration (sigma/cost)
    scores = ei_cost + exploration_bonus
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Clip to avoid extreme values
    scores = np.clip(scores, 0.0, 1e6)
    
    return scores
# EVOLVE-BLOCK-END
