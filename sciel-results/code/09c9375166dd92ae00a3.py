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
    
    # Compute Probability of Improvement (PI) for minimization
    # PI(x) = P(f(x) < f_best)
    # Use a temperature-scaled z to sharpen the acquisition:
    # lower temperature => more decisive (exploitative)
    # higher temperature => smoother (exploratory)
    # Temperature 0.7 balances exploration and exploitation
    temperature = 0.7
    z = (f_best - mu) / sigma
    z_scaled = z / temperature
    pi = norm.cdf(z_scaled)
    
    # Compute Cost-Adjusted PI
    # Use a softer cost penalty: divide by cost^0.5 instead of cost
    # This reduces the aggressive penalty on expensive candidates
    # while still preferring cheaper points
    cost_penalty = np.sqrt(cost)
    pi_cost = pi / cost_penalty
    
    # Ensure scores are finite
    scores = pi_cost
    
    # Optional: Add a small exploration boost for very high uncertainty
    # if we haven't found good points yet, but PI already does this.
    # Let's stick to the pure structural change of EI -> PI.
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Store best scores for potential use
    _acq_state['best_scores'] = scores.copy()
    
    return scores
# EVOLVE-BLOCK-END
