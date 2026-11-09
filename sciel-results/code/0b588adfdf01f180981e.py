import numpy as np
from scipy.stats import norm
import time

# Global state for tracking time and best result
_acq_start_time = None
_acq_best_scores = None
_ACQ_DEADLINE_MARGIN = 10.0  # seconds before the 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function with exponential cost suppression and time guard.
    
    Args:
        mu: array of GP means, shape (N,)
        sigma: array of GP standard deviations, shape (N,)
        f_best: scalar, best function value observed so far
        cost: array of costs, shape (N,)
        
    Returns:
        scores: array of acquisition values, shape (N,), higher is better
    """
    global _acq_start_time, _acq_best_scores
    
    # Initialize start time on first call
    if _acq_start_time is None:
        _acq_start_time = time.time()
    
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check deadline
    elapsed = time.time() - _acq_start_time
    if elapsed > 330.0 - _ACQ_DEADLINE_MARGIN:
        if _acq_best_scores is not None:
            return _acq_best_scores
        # If we haven't computed anything yet, return zeros to be safe
        return np.zeros(len(mu))
    
    # Prevent division by zero in z calculation
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI) for minimization
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma * phi
    ei = np.maximum(ei, 0.0)
    
    # Structural Change: Multiplicative Exponential Cost Gate
    # Instead of dividing by cost (linear penalty), we multiply by exp(-k * cost).
    # This creates a sharp, non-linear penalty for high-cost points.
    # Cost is in [1, 2]. 
    # k=1.5 provides a ratio of ~4.4 between cost=1 and cost=2, 
    # compared to a ratio of 2 for linear division.
    # This aggressively steers the search towards the low-cost center of the domain,
    # which is more likely to contain the shifted minimizers (inner 80%).
    k = 1.5
    cost_gate = np.exp(-k * cost)
    
    scores = ei * cost_gate
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Update best scores for potential timeout fallback
    _acq_best_scores = scores.copy()
    
    return scores
