"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track elapsed time to ensure we finish well within the 300s limit
_START_TIME = time.time()
DEADLINE_LIMIT = 300.0
SAFETY_MARGIN = 10.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining_budget():
    return DEADLINE_LIMIT - _elapsed() - SAFETY_MARGIN

def _expected_improvement(mu, sigma, f_best, xi=0.0):
    """
    Standard Expected Improvement for minimization.
    EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu - xi) / sigma
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def _probability_of_improvement(mu, sigma, f_best):
    """
    Probability of Improvement for minimization: P(f(x) < f_best).
    PI = Phi((f_best - mu) / sigma)
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    return norm.cdf(z)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines Expected Improvement (EI), Upper Confidence Bound (UCB),
    and Probability of Improvement (PI) with cost awareness.
    
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
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Scale factor for normalization (helps when f_best is large vs small)
    f_best_scale = abs(f_best) + 1.0
    
    # 1. Expected Improvement (EI)
    # Use a small xi to slightly favor exploitation as we get closer to the minimum
    xi = 0.01 * abs(f_best)
    ei = _expected_improvement(mu, sigma, f_best, xi=xi)
    
    # Cost-aware EI: divide by cost to prefer cheaper evaluations
    # Use softened cost division to avoid extreme values
    cost_factor = cost + 0.5  # Soften: cost in [1.5, 2.5]
    ei_cost = ei / cost_factor
    
    # 2. UCB for minimization: we want low mu + high uncertainty
    # Adaptive beta: start exploratory, become exploitative as f_best improves
    # When f_best is large (early), explore more; when small (late), exploit more
    beta = 1.5 / (1.0 + 0.5 * f_best_scale)
    ucb = mu - beta * sigma
    # Normalize UCB relative to f_best: how much better than f_best?
    ucb_score = np.maximum(f_best - ucb, 0.0) / f_best_scale
    
    # 3. Probability of Improvement (PI)
    pi = _probability_of_improvement(mu, sigma, f_best)
    
    # 4. Cost-aware exploration: high uncertainty in cheap regions
    # Normalize sigma relative to its range
    sigma_max = np.max(sigma)
    if sigma_max > 1e-9:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Normalize cost to [0, 1] range (lower cost = higher preference)
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range > 1e-9:
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.zeros_like(cost)
    
    # Cost-aware exploration: high sigma, low cost
    cost_exp = sigma_norm * (1.0 - cost_norm)
    
    # Combine with adaptive weights based on progress
    # Early (f_best large): more exploration (UCB, cost_exp)
    # Late (f_best small): more exploitation (EI, PI)
    # f_best_scale = abs(f_best) + 1.0
    # When f_best is large (e.g., 100), f_best_scale = 101, explore_weight should be high
    # When f_best is small (e.g., 0.1), f_best_scale = 1.1, explore_weight should be low
    explore_weight = min(1.0, f_best_scale / 5.0)
    exploit_weight = 1.0 - explore_weight
    
    # Adaptive weights
    w_ei = 0.5 * exploit_weight + 0.3 * explore_weight
    w_ucb = 0.3 * explore_weight + 0.2 * exploit_weight
    w_pi = 0.1
    w_cost_exp = 0.1 * explore_weight
    
    # Normalize EI to be comparable with other terms
    ei_max = np.max(ei_cost)
    if ei_max > 1e-9:
        ei_norm = ei_cost / ei_max
    else:
        ei_norm = np.zeros_like(ei_cost)
    
    # Final score
    score = w_ei * ei_norm + w_ucb * ucb_score + w_pi * pi + w_cost_exp * cost_exp
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
