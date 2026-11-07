"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track start time to ensure we return before 330s
_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement with adaptive
    threshold and cost normalization.
    
    Uses EI with a small epsilon to focus on meaningful improvements,
    combined with an exploration bonus that decays as uncertainty decreases.
    """
    # Check deadline - if close to limit, return a simple fallback
    elapsed = time.time() - _start_time
    if elapsed > 320.0:  # Safety margin before 330s kill
        cost_safe = np.maximum(cost, 1e-8)
        fallback_scores = -mu / cost_safe
        return np.where(np.isfinite(fallback_scores), fallback_scores, 0.0)
    
    # Ensure sigma is safe for division
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Adaptive epsilon: small to focus on real improvements
    epsilon = 0.005
    target = f_best - epsilon
    
    # Calculate z-score
    z = (target - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Expected Improvement: E[max(target - f(x), 0)]
    # EI = (target - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (target - mu) * Phi_z + sigma_safe * phi_z
    
    # Add a small exploration bonus for high uncertainty regions
    # This helps escape local optima early in optimization
    exploration_bonus = 0.1 * sigma_safe / (sigma_safe + 0.1)
    
    # Combine EI and exploration bonus
    combined_score = ei + exploration_bonus
    
    # Normalize by cost to get cost-aware score
    scores = combined_score / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
