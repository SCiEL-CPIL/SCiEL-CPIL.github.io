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
        # Return a safe default: LCB / cost
        mu_safe = np.where(np.isfinite(mu), mu, f_best)
        sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
        cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        kappa = 1.0
        lcb = mu_safe - kappa * sigma_safe
        scores = -lcb / np.maximum(cost_safe, 1e-8)
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
    
    n = len(mu)
    
    # Bayesian Regret Minimization approach:
    # We want to approximate the expected minimum of the function conditioned on the cost budget.
    # This is equivalent to maximizing the probability that a point is the global minimum,
    # weighted by cost efficiency.
    
    # Compute the z-score for each point relative to f_best
    z = (f_best - mu) / sigma
    
    # Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Expected Improvement (EI)
    ei = np.maximum((f_best - mu) * pi + sigma * norm.pdf(z), 0.0)
    
    # Lower Confidence Bound (LCB) - more aggressive exploration
    # Use an adaptive kappa based on the uncertainty
    kappa = 1.0 + 0.5 * np.sqrt(np.mean(sigma**2) + 1e-10)
    lcb = mu - kappa * sigma
    
    # Combine PI and LCB with cost weighting
    # Use a soft-max-like combination to balance exploitation (PI) and exploration (LCB)
    # Normalize PI and LCB to [0, 1] range for combination
    pi_norm = pi / (np.max(pi) + 1e-10)
    lcb_norm = (-lcb) / (np.max(-lcb) + 1e-10)
    
    # Weighted combination: 60% PI, 40% LCB
    combined = 0.6 * pi_norm + 0.4 * lcb_norm
    
    # Cost-aware weighting: use a smooth cost factor
    # Normalize cost to [0, 1]
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + 1e-10
    cost_norm = (cost - cost_min) / cost_range
    
    # Smooth cost factor: penalize high cost but not too aggressively
    # Use a Gaussian-like decay
    cost_factor = np.exp(-0.5 * (cost_norm - 0.3) ** 2 / (0.4 ** 2))
    
    # Final score: combined acquisition * cost_factor
    scores = combined * cost_factor
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
