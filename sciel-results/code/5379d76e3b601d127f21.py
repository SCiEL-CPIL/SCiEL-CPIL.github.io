"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import time

_acq_state = {'start_time': None, 'call_count': 0, 'best_scores': None}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions in objective units
    - sigma: array of shape (2000,), GP standard deviations in objective units
    - f_best: scalar, lowest observed function value
    - cost: array of shape (2000,), evaluation cost for each candidate
    
    Returns:
    - scores: array of shape (2000,), acquisition values (higher = evaluate next)
    """
    # Initialize module-level state
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    _acq_state['call_count'] += 1
    
    # Deadline guard
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 320.0:
        # Return EI/cost as a safe default
        mu_safe = np.where(np.isfinite(mu), mu, f_best)
        sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
        cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        z = (f_best - mu_safe) / np.maximum(sigma_safe, 1e-8)
        ei = np.maximum((f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z), 0.0)
        return ei / np.maximum(cost_safe, 1e-8)
    
    # Ensure proper array types
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Ensure sigma is at least a tiny positive value
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    n = len(mu)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = np.maximum((f_best - mu) * cdf_z + sigma * pdf_z, 0.0)
    
    # Compute Cost-Adjusted EI
    # We want to maximize the expected improvement per unit cost
    ei_cost = ei / cost
    
    # Add a small penalty for very high cost to encourage cheaper evaluations
    # but not too aggressive
    cost_penalty = 0.01 * (cost - np.mean(cost)) / (np.std(cost) + 1e-10)
    scores = ei_cost * (1.0 - cost_penalty)
    
    # Add a small exploration term for early iterations
    n_calls = _acq_state['call_count']
    if n_calls < 8:
        # Early: add more exploration
        exploration_weight = 0.3 * np.exp(-0.3 * n_calls)
        ucb = -mu + 1.5 * sigma
        ucb_cost = ucb / cost
        scores = (1.0 - exploration_weight) * scores + exploration_weight * ucb_cost
    elif n_calls < 15:
        # Middle: moderate exploration
        exploration_weight = 0.15 * np.exp(-0.2 * (n_calls - 8))
        ucb = -mu + 1.0 * sigma
        ucb_cost = ucb / cost
        scores = (1.0 - exploration_weight) * scores + exploration_weight * ucb_cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Store best scores for potential use
    _acq_state['best_scores'] = scores.copy()
    
    return scores
# EVOLVE-BLOCK-END
