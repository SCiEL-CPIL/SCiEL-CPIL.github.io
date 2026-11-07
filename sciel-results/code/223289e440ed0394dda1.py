"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    
    Structural change: Replaces the previous hybrid EI+sigma blend with a
    dynamic Lower Confidence Bound (LCB) that uses an adaptive exploration
    coefficient (kappa) scheduled over the evaluation budget. This approach
    provides a more direct and tunable balance between exploitation (low mu)
    and exploration (high sigma) than the normalized blend, while retaining
    the scheduled exploration-to-exploitation shift.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    global _start_time, _call_count
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    # Deadline guard: return pure exploitation to avoid timeout
    if time.time() - _start_time > 280:
        scores = -mu
        return np.where(np.isfinite(scores), scores, -np.inf)
    
    # Ensure valid inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # --- Adaptive LCB for minimization ---
    # Score = -mu + kappa * sigma
    # kappa is scheduled: high early (exploration), low late (exploitation)
    # 30 total evaluations, 5 initial random, so 25 acquisition-driven steps
    # _call_count goes from 1 to 25 (approximately)
    t = min(_call_count, 25) / 25.0  # 0 to 1 over the budget
    
    # Kappa schedule: starts at 3.5 (aggressive exploration), ends at 1.5 (exploitation)
    # This is inspired by the successful LCB with kappa=3.0 from inspirations,
    # but made adaptive to match the original program's scheduling philosophy.
    kappa = 3.5 - 2.0 * t
    
    # Compute LCB scores: higher is better
    scores = -mu + kappa * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
