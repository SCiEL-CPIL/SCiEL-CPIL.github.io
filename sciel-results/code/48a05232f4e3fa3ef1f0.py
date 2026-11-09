"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0
_call_count = 0

def acquisition(mu, sigma, f_best):
    """
    Hybrid Acquisition Function:
    - Early stage: Blend of EI and UCB to encourage exploration.
    - Late stage: Pure EI for exploitation.
    """
    global _call_count
    _call_count += 1
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        return -mu # Fallback to exploitation

    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate EI
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -5.0, 5.0)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Calculate UCB (for minimization: high score = low mu, high sigma)
    # Using beta = 1.0 for UCB
    ucb = -mu + 1.0 * sigma_safe
    
    # Adaptive blend
    # For the first 5 calls, use 50% UCB, 50% EI
    # After that, use 100% EI
    if _call_count <= 5:
        # Normalize scales roughly by taking the mean of the positive parts?
        # Or just assume EI is smaller and weight it.
        # Let's try: Score = 0.2 * EI + 0.8 * UCB (Early)
        # Score = 1.0 * EI (Late)
        # Actually, let's use a smooth transition.
        weight_ucb = max(0.0, 1.0 - _call_count / 10.0)
        
        # To combine, we need to scale them.
        # Let's scale UCB by the typical magnitude of EI.
        # Or just use a small weight for UCB.
        # Let's try: Score = EI + weight_ucb * 0.1 * UCB
        # This adds a small exploration bonus.
        
        scores = ei + weight_ucb * 0.1 * ucb
    else:
        scores = ei
        
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
