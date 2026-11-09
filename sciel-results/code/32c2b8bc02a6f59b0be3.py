import numpy as np
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
    
    n_calls = _state['n_calls']
    
    # Adaptive Beta for UCB
    # Based on the successful program p76 which scored 1.0885.
    # The diff suggests a change in comments/logic but the core formula might have been tuned.
    # Let's try a slightly more aggressive exploitation schedule or different decay.
    # The previous one was: beta = 1.2 * np.exp(-0.15 * n_calls) + 0.4
    # Let's try: beta = 1.0 * np.exp(-0.2 * n_calls) + 0.3
    
    beta = 1.0 * np.exp(-0.2 * n_calls) + 0.3
    
    # Compute LCB (Lower Confidence Bound) for minimization
    lcb = mu - beta * sigma_safe
    
    # Cost-aware scoring
    scores = -lcb / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
