"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function with a high fixed beta.
    
    The Beale function has a narrow global minimum and broad local minima.
    High exploration (high beta) is crucial to escape local basins early in the search.
    Empirical results show that a fixed high beta (e.g., 5.0) significantly outperforms
    adaptive betas that decay too quickly.
    
    Args:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior std
        f_best: float - best observed value so far
    
    Returns:
        array of shape (2000,) - acquisition scores (higher is better for selection)
    """
    # Deadline guard: Ensure we don't hang the evaluator
    # The function is called 25 times per seed. 3 seeds. Total 75 calls.
    # 330s / 75 = 4.4s per call. We are well within this for N=2000.
    
    # Ensure inputs are numpy arrays and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace non-finite values in mu and sigma with safe defaults
    # If mu is inf, it's likely -inf (good for minimization) or +inf (bad). 
    # GP means should be finite. If not, treat as 0.
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # Ensure sigma is non-negative for safety
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Use a high fixed beta to encourage strong exploration.
    # This is based on the high-performing Inspiration 2.
    beta = 5.0
    
    # LCB for minimization: we want to minimize mu - beta * sigma.
    # The acquisition function returns scores where higher is better.
    # So we return - (mu - beta * sigma) = -mu + beta * sigma.
    scores = -mu + beta * sigma_safe
    
    # Ensure the output is finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
