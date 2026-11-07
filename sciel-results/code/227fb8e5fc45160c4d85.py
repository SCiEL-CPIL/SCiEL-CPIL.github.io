"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global variables for deadline guard
_start_time = None
_deadline = 300.0
_safety_margin = 5.0

def _check_deadline():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    if time.time() > _start_time + _deadline - _safety_margin:
        return True
    return False

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Ensure sigma is positive for division and log
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for Expected Improvement
    # Clip z to avoid extreme values in norm.cdf/pdf which can cause numerical issues
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Expected Improvement (minimization)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * phi_z
    
    # Add a small exploration term to encourage visiting high-uncertainty areas
    # This helps in multi-modal functions where EI might be low if mu > f_best
    exploration_bonus = 0.1 * sigma_safe
    
    # Cost-aware score
    # We want to maximize (Improvement + Exploration) / Cost
    cost_safe = np.maximum(cost, 1e-8)
    
    scores = (ei + exploration_bonus) / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
