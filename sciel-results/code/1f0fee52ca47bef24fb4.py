"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Aware Expected Improvement (CEI) acquisition function.
    
    Strategy:
    Switch from UCB to Expected Improvement (EI). EI is generally more robust 
    for multimodal functions like Hartmann 3D because it explicitly models 
    the probability of improvement over the current best value (f_best). 
    Unlike UCB with a fixed beta, EI naturally balances exploration and 
    exploitation as f_best improves and sigma decreases. Dividing by cost 
    makes it cost-aware.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Fallback: simple greedy exploitation if time is critical
        # Pick the point with the lowest mu, penalized by cost
        scores = -mu / cost
        return np.where(np.isfinite(scores), scores, -np.inf)

    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Z-score for EI
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to prevent numerical issues in CDF/PDF
    # For z > 30, CDF ~ 1, PDF ~ 0
    # For z < -30, CDF ~ 0, PDF ~ 0
    z_clipped = np.clip(z, -30, 30)
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: f_best - mu is the "potential improvement"
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical artifacts might make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    # cost is >= 1.0
    scores = ei / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
