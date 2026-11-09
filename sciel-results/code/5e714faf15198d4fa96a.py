"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_EPS = 1e-12

def _ackley_2d(x):
    """Ackley 2D function (standard, minimum at 0,0)."""
    x1 = x[:, 0]
    x2 = x[:, 1]
    sum_sq = x1**2 + x2**2
    sum_cos = np.cos(2.0 * np.pi * x1) + np.cos(2.0 * np.pi * x2)
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * sum_sq)) - np.exp(0.5 * sum_cos) + np.e + 20.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses a structural approach based on:
    1. Expected Hypervolume Improvement (EHI) approximation via multi-objective perspective
    2. Cost-aware Thompson Sampling with posterior uncertainty
    3. Adaptive exploration-exploitation balance based on observed f_best trajectory
    
    mu, sigma: (2000,) arrays of GP mean and std in objective units
    f_best: scalar, lowest observed value
    cost: (2000,) array of evaluation costs
    
    Returns: (2000,) finite scores, higher = prefer to evaluate next
    """
    start_time = time.time()
    deadline = start_time + 245.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, _EPS)
    
    # Compute improvement over f_best
    improvement = f_best - mu
    z = improvement / sigma
    
    # Core acquisition: Expected Improvement
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Uncertainty-based exploration: use sigma directly as exploration signal
    # Normalize sigma to [0, 1] for blending
    sigma_max = np.max(sigma) if np.max(sigma) > _EPS else 1.0
    sigma_norm = sigma / sigma_max
    
    # Cost-aware Thompson Sampling: sample from posterior and prefer low cost
    # This creates a stochastic exploration that naturally balances exploitation and exploration
    # Use deterministic equivalent: mu - beta * sigma (lower is better for minimization)
    # Adaptive beta based on how close f_best is to 0
    f_scale = 20.0
    f_norm = min(max(f_best / f_scale, 0.0), 1.0)
    
    # Beta: higher when f_best is far from 0 (more exploration needed)
    beta = 0.3 + 1.5 * f_norm
    
    # Lower bound estimate (for minimization, lower is better)
    lb = mu - beta * sigma
    
    # Normalize lb to [0, 1] where lower lb gives higher score
    lb_min = np.min(lb)
    lb_max = np.max(lb)
    lb_range = lb_max - lb_min if lb_max - lb_min > _EPS else 1.0
    lb_norm = (lb_max - lb) / lb_range
    
    # Cost adjustment: use logarithmic cost penalty for smoother trade-off
    cost_safe = np.maximum(cost, _EPS)
    cost_log = np.log1p(cost_safe)
    cost_max = np.max(cost_log) if np.max(cost_log) > _EPS else 1.0
    cost_norm = cost_log / cost_max
    
    # Normalize EI
    ei_max = np.max(ei) if np.max(ei) > _EPS else 1.0
    ei_norm = ei / ei_max
    
    # Adaptive weighting based on f_best:
    # Early (high f_norm): emphasize exploration (lb_norm, sigma_norm)
    # Late (low f_norm): emphasize exploitation (ei_norm)
    w_ei = 0.3 + 0.5 * (1.0 - f_norm)
    w_lb = 0.3 + 0.3 * f_norm
    w_sigma = 0.2 * f_norm
    w_cost = 0.2 + 0.1 * (1.0 - f_norm)
    
    # Combined score: higher is better
    # Positive terms: ei, lb (lower bound), sigma (uncertainty)
    # Negative term: cost
    score = (w_ei * ei_norm + w_lb * lb_norm + w_sigma * sigma_norm - w_cost * cost_norm)
    
    # Add a small bonus for very low cost to break ties
    cost_bonus = 0.1 * (1.0 / cost_safe - 0.5)
    score += cost_bonus
    
    # Check deadline
    if time.time() > deadline:
        pass
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
