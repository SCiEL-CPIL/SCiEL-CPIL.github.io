import numpy as np
from scipy.stats import norm
import time

# Global state for adaptive parameter tuning
_state = {
    'best_f': np.inf,
    'f_history': [],
    'n_calls': 0,
    'start_time': None,
    'cost_mean': 1.0,
    'cost_std': 0.1
}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement with cost normalization.
    
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
    if elapsed > 290:
        sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
        cost_safe = np.where(cost > 1e-9, cost, 1e-9)
        # Fast EI/cost fallback
        z = np.clip((f_best - mu) / sigma_safe, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Handle edge cases
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    cost_safe = np.where(cost > 1e-9, cost, 1e-9)
    
    # Track progress
    if f_best < _state['best_f']:
        _state['best_f'] = f_best
    
    # Update cost statistics for adaptive normalization
    cost_mean = np.mean(cost_safe)
    cost_std = max(np.std(cost_safe), 1e-9)
    _state['cost_mean'] = 0.9 * _state['cost_mean'] + 0.1 * cost_mean
    _state['cost_std'] = 0.9 * _state['cost_std'] + 0.1 * cost_std
    
    n_calls = _state['n_calls']
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    phi = norm.pdf(z_clipped)
    cdf = norm.cdf(z_clipped)
    
    ei = (f_best - mu) * cdf + sigma_safe * phi
    
    # Cost-aware normalization: divide by cost to get "improvement per unit cost"
    # This naturally prefers cheaper evaluations when EI is similar
    ei_per_cost = ei / cost_safe
    
    # Add a small exploration bonus for high-uncertainty regions
    # Weighted by inverse cost to keep it cost-aware
    exploration_weight = 0.1 / cost_safe
    exploration = sigma_safe * exploration_weight
    
    scores = ei_per_cost + exploration
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
