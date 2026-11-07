"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_start_time = time.time()
_DEADLINE_SECONDS = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Adaptive Lower Confidence Bound (LCB) for minimization.
    Structural change from EI: uses adaptive beta that decreases as f_best approaches 0,
    shifting from exploration to exploitation.
    
    mu: (N,) array of posterior means
    sigma: (N,) array of posterior std devs  
    f_best: float, best observed value so far
    Returns: (N,) array of scores (higher = better to evaluate next)
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE_SECONDS:
        # Return a safe fallback: pure exploitation (lowest mu)
        sigma_safe = np.maximum(sigma, 1e-8)
        return np.where(np.isfinite(mu), -mu, 0.0)
    
    # Ensure sigma is positive and numerically stable
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Adaptive beta: high when f_best is large (early), low when f_best is small (late)
    # For sphere 1D on [-5,5], f_best ranges from ~25 down to 0
    # Use a smooth transition
    f_best_clipped = max(f_best, 1e-6)
    
    # Beta adapts: starts around 3.0, decreases to ~0.5 as f_best -> 0
    # Using a power law: beta = base + scale * (f_best / f_ref)^alpha
    f_ref = 25.0  # approximate initial f_best for x in [-5,5]
    ratio = min(f_best_clipped / f_ref, 1.0)
    beta = 0.5 + 2.5 * (ratio ** 0.5)
    
    # Clip beta to reasonable range
    beta = np.clip(beta, 0.3, 4.0)
    
    # LCB score: we want to find where f is likely low
    # score = -mu + beta * sigma (higher = more promising for minimization)
    score = -mu + beta * sigma_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Optional: add a tiny sigma-based exploration bonus to break ties
    # This helps in flat regions where mu is similar
    score = score + 0.1 * sigma_safe
    
    return score
# EVOLVE-BLOCK-END
