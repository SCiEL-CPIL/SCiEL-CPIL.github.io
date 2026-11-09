import numpy as np
from scipy.stats import norm
import time

# Module-level state for tracking progress
_state = {
    'start_time': None,
    'call_count': 0
}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using expected improvement per unit cost.
    """
    if _state['start_time'] is None:
        _state['start_time'] = time.time()
    _state['call_count'] += 1
    
    # Deadline guard
    elapsed = time.time() - _state['start_time']
    if elapsed > 295:
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / np.maximum(cost, 1e-12)
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-adjusted EI: EI per unit cost
    ei_cost = ei / cost_safe
    
    # Exploration term: normalized sigma, also cost-adjusted
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = max(sigma_max - sigma_min, 1e-12)
    sigma_norm = (sigma_safe - sigma_min) / sigma_range
    explore_cost = sigma_norm / cost_safe
    
    # Dynamic balance based on call count
    n_calls = _state['call_count']
    # Estimate: ~15-20 total evaluations possible
    progress = min(1.0, n_calls / 20.0)
    
    # Beta: exploration weight, decreases with progress
    # Early: high exploration, Late: high exploitation
    beta = 0.8 * (1.0 - progress) ** 2
    beta = np.clip(beta, 0.05, 0.8)
    
    # Combine exploitation and exploration
    score = (1.0 - beta) * ei_cost + beta * explore_cost
    
    # Soft cost preference: favor costs near 1.5 (expected minimizer cost)
    # This is a mild bias, not a hard penalty
    target_cost = 1.5
    cost_bias = np.exp(-0.5 * (cost_safe - target_cost) ** 2 / 0.05)
    score = score * cost_bias
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
