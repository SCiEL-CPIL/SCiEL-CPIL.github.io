"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE_SECONDS = 300.0  # 300s limit per evaluation, leave margin

def _check_deadline():
    """Return True if we should stop and return current best."""
    return (time.time() - _START_TIME) > (_DEADLINE_SECONDS - 5.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a deterministic Worst-Case Improvement (WCI) strategy, 
    defined as max(0, f_best - mu - k*sigma), weighted by cost.
    This shifts the objective from expected improvement to robust 
    minimax regret, penalizing high uncertainty regions.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions in objective units
    sigma : array of shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Best (lowest) observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition values (higher = better to evaluate)
    """
    # Ensure inputs are numpy arrays and handle edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clamp sigma to avoid division by zero or negative values
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Worst-Case Improvement (WCI)
    # We use a conservative exploration constant k.
    # A smaller k favors exploitation (confident improvements).
    # A larger k favors exploration (regions with high potential but high uncertainty).
    # For Shekel 4d, which is multimodal but relatively smooth, a moderate k is appropriate.
    # The previous EI approach effectively had a variable "k" that decreased with confidence.
    # Here we use a fixed k to enforce robustness.
    k = 0.5  # Conservative exploration: we require a 0.5 sigma margin for worst-case improvement
    
    wci = np.maximum(0.0, f_best - mu - k * sigma_safe)
    
    # Cost-normalized score
    score = wci / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
