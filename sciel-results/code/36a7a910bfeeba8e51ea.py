"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_acq_state = {'start_time': None, 'call_count': 0}

def acquisition(mu, sigma, f_best, cost):
    """
    Bayesian Regret Minimization (BRM) acquisition for cost-aware BO.
    
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
        # Safe fallback: PI / cost
        mu_safe = np.where(np.isfinite(mu), mu, f_best)
        sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
        cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        z_safe = (f_best - mu_safe) / np.maximum(sigma_safe, 1e-8)
        pi_safe = norm.cdf(z_safe)
        scores = pi_safe / np.maximum(cost_safe, 1e-8)
        return np.where(np.isfinite(scores), scores, 0.0)
    
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
    
    # BRM: Approximate the probability that each candidate is the global minimum
    # Given the GP, the probability that x is the minimum among all candidates
    # can be approximated by the probability that f(x) < f(x') for all other x'.
    
    # A practical approximation: use the "probability of being the minimum" 
    # which relates to the CDF of f(x) being lower than f_best, weighted by 
    # the uncertainty (high sigma means more likely to be the true minimum).
    
    # Key BRM insight: P(x is min) ~ P(f(x) < f_best) * (1 + uncertainty bonus)
    # But more precisely, we want to maximize E[min(f(x))] which is:
    # E[min] = integral of min over all joint samples
    
    # Practical BRM approximation:
    # Score(x) = P(f(x) < f_best) / cost(x) * exploration_bonus(x)
    # where exploration_bonus accounts for the fact that uncertain points
    # have a higher chance of being the true global minimum.
    
    # Compute probability of improvement (PI)
    z = (f_best - mu) / sigma
    pi = norm.cdf(z)
    
    # Exploration bonus: points with higher sigma have more chance of being
    # the true minimum. Use the PDF of the improvement distribution as a proxy.
    # The density of the improvement at zero indicates how "close" we are to
    # the boundary of improvement, which relates to the probability of being
    # the new minimum.
    improvement_density = norm.pdf(z)
    
    # BRM score: combine PI (exploitation) with uncertainty (exploration)
    # The term (1 + c * sigma) weights uncertain points higher
    # This is motivated by: P(x is global min) increases with sigma because
    # the true value could be much lower than the mean
    
    # Adaptive exploration coefficient based on the budget
    # As we spend more budget, we should explore less
    total_cost_spent = np.sum(cost)  # This is approximate; we don't track actual spent
    # Use the mean sigma as a proxy for remaining exploration need
    mean_sigma = np.mean(sigma)
    
    # Exploration coefficient: higher when overall uncertainty is high
    c_explore = 0.5 + 0.5 * np.clip(mean_sigma / (np.std(mu) + 1e-10), 0, 1)
    
    # BRM acquisition: PI * (1 + c_explore * normalized_sigma) / cost
    # Normalize sigma to [0, 1] for the exploration bonus
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    sigma_range = sigma_max - sigma_min + 1e-10
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # The BRM score: probability of being the minimum, weighted by cost
    # P(min) approximation: pi * (1 + c_explore * sigma_norm)
    brm_score = pi * (1.0 + c_explore * sigma_norm)
    
    # Cost-aware: divide by cost to get probability per unit cost
    scores = brm_score / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Optional: add a small exploration floor for very uncertain regions
    # to avoid completely ignoring high-sigma points with low PI
    high_uncertainty_mask = sigma_norm > 0.9
    if np.any(high_uncertainty_mask):
        exploration_floor = np.max(scores) * 0.05
        scores[high_uncertainty_mask] = np.maximum(scores[high_uncertainty_mask], exploration_floor)
    
    return scores
# EVOLVE-BLOCK-END
