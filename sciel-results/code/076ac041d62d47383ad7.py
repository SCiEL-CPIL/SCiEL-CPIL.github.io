import numpy as np
from scipy.stats import norm
import time

# Global state for adaptive parameter tuning
_state = {
    'best_f': np.inf,
    'f_history': [],
    'n_calls': 0,
    'start_time': None
}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Adaptive Lower Confidence Bound (LCB/UCB)
    for minimization.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - best observed function value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = prefer)
    """
    global _state
    
    # Initialize state
    if _state['start_time'] is None:
        _state['start_time'] = time.time()
    
    _state['n_calls'] += 1
    _state['f_history'].append(f_best)
    
    elapsed = time.time() - _state['start_time']
    
    # Safety: if we're close to timeout, use a fast fallback
    if elapsed > 280:
        sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
        cost_safe = np.where(cost > 1e-9, cost, 1e-9)
        # Fast UCB fallback
        beta = 1.0
        lcb = mu - beta * sigma_safe
        scores = -lcb / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Handle edge cases
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    cost_safe = np.where(cost > 1e-9, cost, 1e-9)
    
    # Track progress
    if f_best < _state['best_f']:
        _state['best_f'] = f_best
    
    # Adaptive Beta for UCB
    # Start with high exploration, decay to exploitation
    # Using a hyperbolic decay or log-based decay
    n_calls = _state['n_calls']
    
    # Scale beta based on the scale of the function values to keep exploration meaningful
    # If f_best is large, we can afford more exploration? 
    # Standard UCB: beta = sqrt(log(1 + n))
    # Let's use a decay that starts around 2.0 and drops to 0.5
    # beta = 2.0 * (0.95 ** n_calls) + 0.5
    
    # Better: Beta based on percentile of sigma to ensure exploration is relative to uncertainty
    # But UCB is defined as mu - beta * sigma.
    # Let's try a fixed schedule that is known to work well for 4D Powell
    # The total budget is ~30 cost. Initial 5 points cost ~5*1.5 = 7.5. Remaining ~22.5.
    # We will have roughly 15-20 evaluations.
    
    # Adaptive beta:
    # Early: explore widely. Late: exploit.
    # beta = 1.5 * np.exp(-0.1 * n_calls) + 0.5
    
    # Let's use a beta that adapts to the current uncertainty level
    # If sigma is high everywhere, we need a smaller beta to avoid noise?
    # No, if sigma is high, we WANT to explore.
    
    # Let's stick to a simple, robust decay.
    beta = 1.2 * np.exp(-0.15 * n_calls) + 0.4
    
    # Compute LCB (Lower Confidence Bound) for minimization
    # We want to minimize mu - beta * sigma
    # Score should be high for low LCB.
    # So Score = - (mu - beta * sigma) / cost
    lcb = mu - beta * sigma_safe
    
    # Cost-aware scoring
    # Divide by cost. 
    # To avoid dividing by very small costs (which are ~1.0), we just divide.
    # Cost is in [1, 2].
    scores = -lcb / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
