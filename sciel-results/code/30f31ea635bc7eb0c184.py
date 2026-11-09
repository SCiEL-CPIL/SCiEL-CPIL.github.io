# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time for deadline guard
_START_TIME = time.time()
DEADLINE_MARGIN = 30.0  # Safety margin before 330s limit

def _elapsed():
    return time.time() - _START_TIME

def _expected_improvement(mu, sigma, f_best, xi=0.0):
    """Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def _probability_of_improvement(mu, sigma, f_best, xi=0.0):
    """P(f(x) < f_best - xi) for minimization."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu - xi) / sigma_safe
    return norm.cdf(z)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Args:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value
        cost: (2000,) array of evaluation costs (1 + mean(x_scaled))
    
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're running low on time, return simple EI/cost quickly
    if _elapsed() > DEADLINE_MARGIN:
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / (cost + 1e-12)
        return np.where(np.isfinite(score), score, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Adaptive parameters based on progress
    f_scale = abs(f_best)
    if f_scale > 10.0:
        # Early phase: strong exploitation
        xi = 0.0
        beta = 2.5
        w_ei = 0.60
        w_ucb = 0.15
        w_pi = 0.05
        w_cost_exp = 0.20
    elif f_scale > 1.0:
        # Middle phase: balanced
        xi = 0.01
        beta = 2.0
        w_ei = 0.45
        w_ucb = 0.20
        w_pi = 0.10
        w_cost_exp = 0.25
    else:
        # Late phase: exploration focused, rely on PI for smoothness
        xi = 0.05
        beta = 1.5
        w_ei = 0.30
        w_ucb = 0.20
        w_pi = 0.20
        w_cost_exp = 0.30
    
    # 1. Expected Improvement per unit cost
    ei = _expected_improvement(mu, sigma_safe, f_best, xi=xi)
    ei_cost = ei / (cost + 1e-12)
    
    # 2. UCB for minimization: mu - beta*sigma, normalized
    ucb = mu - beta * sigma_safe
    ucb_improve = np.maximum(f_best - ucb, 0.0)
    ucb_norm = ucb_improve / (abs(f_best) + 1.0)
    
    # 3. Probability of Improvement (smooth, no cliff)
    pi = _probability_of_improvement(mu, sigma_safe, f_best, xi=xi)
    # Normalize PI to [0, 1] scale comparable to others
    pi_norm = pi
    
    # 4. Cost-efficiency of exploration: high sigma / low cost
    cost_exp = sigma_safe / (cost + 1e-12)
    cost_exp_max = np.max(cost_exp) if len(cost_exp) > 0 else 1.0
    cost_exp_norm = cost_exp / (cost_exp_max + 1e-12)
    
    # Combine with adaptive weights
    score = w_ei * ei_cost + w_ucb * ucb_norm + w_pi * pi_norm + w_cost_exp * cost_exp_norm
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
