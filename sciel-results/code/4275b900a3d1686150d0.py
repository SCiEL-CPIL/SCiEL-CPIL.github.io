import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track elapsed time to avoid timeout
_start_time = time.time()
DEADLINE = 320.0  # Safety margin before 330s limit

def _check_deadline():
    return (time.time() - _start_time) > DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Rastrigin 2D.
    
    Uses Expected Improvement per unit cost (cost-aware EI) as the primary
    acquisition, combined with a small exploration bonus for robustness.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x scaled to [0,1]).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Check deadline - if we're running out of time, return a safe fallback
    if _check_deadline():
        # Return scores that prioritize low-cost points near current best
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-9)
        return 1.0 / cost_safe
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Expected Improvement for minimization:
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # We want to minimize f, so improvement is f_best - f(x) when f(x) < f_best
    
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative (it should be, but clip for safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI: divide by cost to prefer cheaper evaluations
    # Use a power of cost to tune the tradeoff
    cost_aware_ei = ei / cost_safe
    
    # Add a small exploration bonus based on sigma to encourage exploring
    # uncertain regions, scaled by inverse cost
    exploration_bonus = 0.3 * sigma_safe / cost_safe
    
    # Combine
    score = cost_aware_ei + exploration_bonus
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
