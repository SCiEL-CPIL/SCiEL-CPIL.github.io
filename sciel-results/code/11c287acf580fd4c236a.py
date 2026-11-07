"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses the Lower Confidence Bound (LCB) strategy for minimization.
    The score is calculated as beta * sigma - mu, encouraging selection of
    points with high uncertainty (sigma) or low mean values (mu).
    
    An increased exploration coefficient (beta=3.8) is used to better handle
    the multimodal nature of the Hartmann 6D function within the tight
    30-evaluation budget, based on previous findings that higher exploration
    yields better global coverage.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Safety margin: If the total time exceeds 280s, we return a simplified score
    # to ensure we don't risk a timeout in the evaluator's loop.
    if time.time() - _start_time > 280:
        # Fall back to a pure exploitation strategy (minimize mu)
        scores = -mu
        return np.where(np.isfinite(scores), scores, -np.inf)

    # Ensure sigma is positive to avoid division by zero or invalid values
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # LCB for minimization:
    # Score = beta * sigma - mu
    # Increased beta from 3.0 to 3.8 to enhance exploration in early/mid stages
    # and prevent premature convergence in local optima.
    beta = 3.8
    
    scores = beta * sigma_safe - mu
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
