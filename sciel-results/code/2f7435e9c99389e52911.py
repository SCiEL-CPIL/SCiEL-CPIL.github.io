"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for time tracking and evaluation count
_start_time = time.time()
_eval_count = 0

def acquisition(mu, sigma, f_best):
    """
    Kriging Believer acquisition function.
    Maps the posterior to a deterministic surrogate objective (Lower Confidence Bound).
    """
    global _eval_count, _start_time
    
    # Deadline guard: return a safe constant if time is running out
    elapsed = time.time() - _start_time
    if elapsed > 320:  # Safety margin before 330s limit
        return np.zeros_like(mu)
        
    _eval_count += 1
    
    # Kriging Believer: evaluate the surrogate at the lower confidence bound
    # y_kb = mu - k * sigma
    # We want to maximize the score, which corresponds to minimizing y_kb.
    # So score = -y_kb = -mu + k * sigma
    
    # Adaptive k:
    # Start with moderate exploration, decay to exploitation.
    # Total budget is 30 evaluations.
    # Early on (steps 1-10), k is higher to find promising basins.
    # Later (steps 11-30), k decreases to refine the minimum.
    
    # Simple decay schedule: k goes from 2.0 down to 0.5
    # Using a smooth decay based on evaluation count
    # At eval 1: k ~ 2.0
    # At eval 30: k ~ 0.5
    # Linear decay: k = 2.0 - (eval-1) * (1.5 / 29)
    if _eval_count <= 1:
        k = 2.0
    else:
        progress = (_eval_count - 1) / 29.0
        k = 2.0 - 1.5 * progress
        if k < 0.5:
            k = 0.5
            
    # Ensure sigma is non-negative for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Kriging Believer score
    # Higher score means more promising (lower predicted value with uncertainty discount)
    score = -mu + k * sigma_safe
    
    return score
# EVOLVE-BLOCK-END
