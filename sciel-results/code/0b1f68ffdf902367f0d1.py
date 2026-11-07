"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_acq_state = {'start_time': None, 'call_count': 0}

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
        # Return PI/cost as a safe default
        mu_safe = np.where(np.isfinite(mu), mu, f_best)
        sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
        cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        z = (f_best - mu_safe) / np.maximum(sigma_safe, 1e-8)
        pi = norm.cdf(z)
        return pi / np.maximum(cost_safe, 1e-8)
    
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
    z = (f_best - mu) / sigma
    pi = norm.cdf(z)
    
    # Compute Lower Confidence Bound (LCB) for minimization
    # LCB(x) = mu(x) - kappa * sigma(x)
    # We want to maximize -LCB / cost
    # Adaptive kappa: start with moderate exploration, reduce as we find better solutions
    # The relative gap between f_best and the GP's best prediction indicates how much more we can improve
    best_mu = np.min(mu)
    gap = f_best - best_mu
    if gap > 0:
        # More room to improve -> more exploration
        kappa = 0.5 + 0.5 * np.tanh(gap / (np.std(mu) + 1e-8))
    else:
        # Little room to improve -> less exploration
        kappa = 0.5
        
    lcb = mu - kappa * sigma
    lcb_score = -lcb / cost
    
    # Combine PI and LCB scores
    # Normalize both to [0, 1] for a balanced combination
    pi_max = np.max(pi)
    if pi_max > 1e-10:
        pi_norm = pi / pi_max
    else:
        pi_norm = pi
    
    lcb_max = np.max(lcb_score)
    if lcb_max > 1e-10:
        lcb_norm = lcb_score / lcb_max
    else:
        lcb_norm = lcb_score
    
    # Weighted combination: 50% PI, 50% LCB
    # PI provides robust exploitation, LCB provides exploration
    combined = 0.5 * pi_norm + 0.5 * lcb_norm
    
    # Cost-aware weighting: use a smooth cost factor
    # Normalize cost to [0, 1] range
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + 1e-10
    cost_norm = (cost - cost_min) / cost_range
    
    # Smooth cost factor: penalize high cost but not too aggressively
    # Use a Gaussian-like decay centered at lower cost values
    # This encourages cheaper evaluations while still allowing expensive ones if promising
    cost_factor = np.exp(-0.5 * (cost_norm - 0.2) ** 2 / (0.5 ** 2))
    
    # Final score: combined acquisition * cost_factor
    scores = combined * cost_factor
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
