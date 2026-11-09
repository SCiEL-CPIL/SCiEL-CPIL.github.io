# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for safety
_START_TIME = time.time()
_TIME_LIMIT = 300  # Seconds

def _check_time():
    """Check if we are close to the time limit."""
    return (time.time() - _START_TIME) > (_TIME_LIMIT - 5)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) with
    dynamically increasing exploration constant.
    
    For minimization:
    - UCB score = -(mu - kappa * sigma) = -mu + kappa * sigma
    - kappa scales with sqrt(n_evals) to balance exploration/exploitation
    - n_evals is estimated from the number of points where sigma is very small
      (indicating previously evaluated points) plus a base offset
    
    The negative sign makes higher scores correspond to lower predicted values
    (exploitation) with a bonus for uncertainty (exploration).
    """
    # Safety check: if we are running out of time, return a safe fallback
    if _check_time():
        # Fallback: return -mu (pure exploitation) to ensure we finish
        return -mu

    # Estimate number of evaluations so far
    # Points with very small sigma are likely near previously evaluated points
    # We use a heuristic: count points where sigma is below a threshold
    # This gives a rough estimate of how many evaluations have been done
    sigma_threshold = 1e-3
    n_low_sigma = np.sum(sigma < sigma_threshold)
    # Base offset of 5 (initial points) plus estimated additional evaluations
    # The number of low-sigma points correlates with proximity to evaluated points
    n_evals_est = max(5, 5 + int(n_low_sigma / 50))  # heuristic scaling
    
    # Dynamic kappa: starts higher for exploration, decreases as we learn more
    # kappa = base + decay * sqrt(n_evals)
    # For small n_evals, kappa is larger (more exploration)
    # For large n_evals, kappa is smaller (more exploitation)
    # Using a form that transitions from exploration to exploitation
    kappa = 1.0 + 0.5 * np.sqrt(n_evals_est)
    
    # UCB for minimization: we want to minimize mu - kappa*sigma
    # So the acquisition score (higher is better) is: -(mu - kappa*sigma) = -mu + kappa*sigma
    score = -mu + kappa * sigma
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
