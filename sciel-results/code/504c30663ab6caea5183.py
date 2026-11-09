# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_START_TIME = time.time()
DEADLINE_LIMIT = 330.0
SAFETY_MARGIN = 35.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining_budget():
    return DEADLINE_LIMIT - _elapsed() - SAFETY_MARGIN

def _expected_improvement(mu, sigma, f_best, xi=0.0):
    """Standard EI for minimization with exploitation bonus xi."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def _acquisition(mu, sigma, f_best, cost):
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
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Cost normalization: costs are in [1, 2] for rosenbrock on [-2,2]^2
    # cost = 1 + mean(x scaled to [0,1]), so cost in [1, 2]
    # Normalize cost to [0, 1] range for better scaling
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range > 1e-9:
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.zeros_like(cost)
    
    # 1. Expected Improvement (with small xi for exploitation)
    # Use xi that scales with f_best to balance exploration/exploitation
    xi = 0.01 * abs(f_best)
    ei = _expected_improvement(mu, sigma, f_best, xi=xi)
    
    # Cost-aware EI: divide by cost to prefer cheaper evaluations
    # Use a softened cost division to avoid extreme values
    cost_factor = cost + 0.5  # Soften: cost in [1.5, 2.5]
    ei_cost = ei / cost_factor
    
    # 2. UCB for minimization: we want low mu + high uncertainty
    # Adaptive beta: start exploratory, become exploitative as f_best improves
    # Normalize f_best for adaptive beta
    f_best_scale = abs(f_best) + 1.0
    # Beta decreases as we get closer to the minimum
    beta = 1.5 / (1.0 + 0.5 * f_best_scale)
    ucb = mu - beta * sigma
    # Normalize UCB relative to f_best
    ucb_score = (f_best - ucb) / f_best_scale
    ucb_score = np.maximum(ucb_score, 0.0)
    
    # 3. Probability of Improvement
    z = (f_best - mu) / sigma
    pi = norm.cdf(z)
    
    # 4. Pure uncertainty (for exploration in cheap regions)
    # Normalize sigma relative to its range
    sigma_max = np.max(sigma)
    if sigma_max > 1e-9:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # 5. Cost-aware exploration: high uncertainty in cheap regions
    cost_exp = sigma_norm * (1.0 - cost_norm)  # High sigma, low cost
    
    # Combine with adaptive weights based on progress
    # Early: more exploration (UCB, cost_exp)
    # Late: more exploitation (EI, PI)
    progress = min(f_best_scale / 10.0, 1.0)  # 0 at start, 1 when f_best < 10
    # Actually, rosenbrock min is 0, so f_best starts high and decreases
    # Use f_best directly: when f_best is large, explore more
    explore_weight = min(1.0, f_best_scale / 5.0)
    exploit_weight = 1.0 - explore_weight
    
    # Adaptive weights
    w_ei = 0.5 * exploit_weight + 0.3 * explore_weight
    w_ucb = 0.3 * explore_weight + 0.2 * exploit_weight
    w_pi = 0.15 * exploit_weight + 0.05 * explore_weight
    w_cost_exp = 0.15 * explore_weight + 0.05 * exploit_weight
    
    # Ensure weights sum to 1
    total_w = w_ei + w_ucb + w_pi + w_cost_exp
    w_ei /= total_w
    w_ucb /= total_w
    w_pi /= total_w
    w_cost_exp /= total_w
    
    # Final score
    score = (w_ei * ei_cost + w_ucb * ucb_score + w_pi * pi + w_cost_exp * cost_exp)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

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
    # Deadline guard
    if _remaining_budget() < 5.0:
        # Return a simple fallback to avoid timeout
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma = np.maximum(sigma, 1e-12)
        # Simple EI/cost
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / (cost + 1.0)
        return np.where(np.isfinite(score), score, 0.0)
    
    return _acquisition(mu, sigma, f_best, cost)
# EVOLVE-BLOCK-END
