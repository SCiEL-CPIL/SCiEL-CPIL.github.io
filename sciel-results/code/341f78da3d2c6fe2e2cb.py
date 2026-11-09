# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_START_TIME = time.time()
DEADLINE_LIMIT = 330.0
SAFETY_MARGIN = 20.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining_budget():
    return DEADLINE_LIMIT - _elapsed() - SAFETY_MARGIN

def _expected_improvement(mu, sigma, f_best, xi=0.0):
    """Standard EI for minimization with exploitation bonus xi."""
    sigma_safe = np.maximum(sigma, 1e-9)
    improvement = f_best - mu - xi
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def _ucb(mu, sigma, f_best, beta=2.0):
    """UCB for minimization: lower predicted value + uncertainty exploration."""
    sigma_safe = np.maximum(sigma, 1e-9)
    return mu - beta * sigma_safe

def _probability_of_improvement(mu, sigma, f_best, xi=0.0):
    """PI for minimization: P(f(x) < f_best - xi)."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu - xi) / sigma_safe
    return norm.cdf(z)

def _cost_adjusted_ei(mu, sigma, f_best, cost, xi=0.0):
    """EI normalized by cost."""
    ei = _expected_improvement(mu, sigma, f_best, xi)
    cost_safe = np.maximum(cost, 1e-9)
    return ei / cost_safe

def _ucb_cost_aware(mu, sigma, f_best, cost, beta=2.0):
    """UCB normalized by cost."""
    ucb = _ucb(mu, sigma, f_best, beta)
    # For minimization, lower UCB is better. We want high score.
    # Score = (f_best - ucb) / cost
    cost_safe = np.maximum(cost, 1e-9)
    improvement = f_best - ucb
    return improvement / cost_safe

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Args:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check remaining budget - if too little, return a simple score
    if _remaining_budget() < 5.0:
        # Fallback: simple EI/cost
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        cost_safe = np.maximum(cost, 1e-9)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute multiple acquisition components
    
    # 1. Cost-adjusted EI (primary exploitation signal)
    ei_cost = _cost_adjusted_ei(mu, sigma_safe, f_best, cost_safe, xi=0.01)
    
    # 2. Cost-adjusted UCB (exploration signal)
    ucb_cost = _ucb_cost_aware(mu, sigma_safe, f_best, cost_safe, beta=1.5)
    
    # 3. Probability of Improvement (gating signal)
    pi = _probability_of_improvement(mu, sigma_safe, f_best, xi=0.0)
    
    # 4. Pure uncertainty per cost (raw exploration)
    uncertainty_cost = sigma_safe / cost_safe
    
    # Adaptive weighting based on f_best
    # When f_best is large (early on), explore more
    # When f_best is small (later), exploit more
    f_best_abs = abs(f_best)
    
    # Weight that shifts from exploration to exploitation as we converge
    # f_best starts large, decreases toward 0
    exploit_weight = 1.0 / (1.0 + f_best_abs)
    explore_weight = 1.0 - exploit_weight
    
    # Dynamic weights
    w_ei = 0.5 * exploit_weight + 0.3  # Higher when exploiting
    w_ucb = 0.3 * explore_weight + 0.2  # Higher when exploring
    w_pi = 0.1
    w_unc = 0.1 * explore_weight
    
    # Normalize weights to sum to 1
    w_sum = w_ei + w_ucb + w_pi + w_unc
    w_ei /= w_sum
    w_ucb /= w_sum
    w_pi /= w_sum
    w_unc /= w_sum
    
    # Normalize each component to [0, 1] range for fair combination
    ei_cost_norm = ei_cost / (np.max(ei_cost) + 1e-9)
    ucb_cost_norm = np.maximum(ucb_cost, 0.0) / (np.max(np.maximum(ucb_cost, 0.0)) + 1e-9)
    pi_norm = pi  # Already in [0, 1]
    uncertainty_cost_norm = uncertainty_cost / (np.max(uncertainty_cost) + 1e-9)
    
    # Combined score
    score = (w_ei * ei_cost_norm + w_ucb * ucb_cost_norm + 
             w_pi * pi_norm + w_unc * uncertainty_cost_norm)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
