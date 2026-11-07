"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = 320.0  # seconds (330s limit, 10s safety margin)
_START_TIME = None


def _ensure_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME


def _check_deadline():
    """Return True if we should stop (deadline exceeded)."""
    return (time.time() - _START_TIME) > _DEADLINE


def _ei_score(mu, sigma, f_best):
    """Expected Improvement."""
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _ucb_score(mu, sigma, beta):
    """Upper Confidence Bound (for minimization: lower is better, so we use lower bound)."""
    # For minimization, we want to explore where the lower bound is high
    # UCB for minimization: mu - beta * sigma (lower confidence bound)
    # We want to maximize this, i.e., find regions where even the lower bound is high
    # Actually for minimization, we explore where the true value could be low
    # So we use: mu + beta * sigma (upper bound) and minimize it
    # But since we return higher = better, we negate: -(mu + beta * sigma)
    # Wait, let me think again.
    # We want to find x that minimizes f(x).
    # UCB for minimization: we explore where the lower bound of f(x) is as high as possible? No.
    # We explore where the true f(x) could be low.
    # Lower bound: mu - beta * sigma. We want this to be as low as possible.
    # So score = -(mu - beta * sigma) = -mu + beta * sigma
    # Higher score = more likely to be a good (low) point.
    return -mu + beta * sigma


def _pi_score(mu, sigma, f_best, cost):
    """Probability of Improvement per unit cost."""
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    return pi / (cost + 1e-9)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - sigma: array of shape (2000,), GP standard deviation predictions
    - f_best: float, lowest observed value
    - cost: array of shape (2000,), evaluation costs
    
    Returns:
    - array of shape (2000,), acquisition scores (higher = evaluate next)
    """
    _ensure_start()
    
    # Check deadline
    if _check_deadline():
        # Return a simple, fast score to avoid timeout
        return _ei_score(mu, sigma, f_best) / (cost + 1e-9)
    
    n = len(mu)
    
    # Progress-based blending: early = more exploration (UCB), late = more exploitation (EI)
    # Estimate progress from cost: initial cost ~5 * mean_cost, max cost = 30
    # Mean cost is roughly 1 + 0.5 = 1.5 for uniform points in [0,1]^2
    # Initial cost ~ 7.5, so progress = (total_cost - 7.5) / (30 - 7.5)
    # But we don't have total_cost directly. We can estimate from the cost array.
    # Actually, the cost array is for the 2000 candidates, not the total spent.
    # Let's use a simple heuristic: blend based on f_best relative to known minimum.
    # Known minimum is -78.3323. If f_best is close to min, we're late.
    
    # Alternative: use the number of evaluations inferred from cost.
    # But we don't have that info. Let's use a fixed blend or a simple heuristic.
    
    # Simple approach: blend 50% EI and 50% UCB with adaptive beta
    # Beta decreases as we get closer to the optimum
    
    # Estimate progress: assume initial f_best is around 0-10, optimum is -78.33
    # progress = (f_best - f_min) / (f_initial - f_min)
    # But we don't know f_initial. Let's use a fixed schedule.
    
    # Use a moderate blend: 60% EI, 40% UCB with beta that adapts to sigma
    # Beta should be larger when sigma is large (more uncertainty)
    
    # Compute EI score
    ei = _ei_score(mu, sigma, f_best)
    ei_cost = ei / (cost + 1e-9)
    
    # Compute UCB score with adaptive beta
    # Beta should be larger early, smaller late
    # Use a heuristic: beta = 2.0 * (sigma / (sigma + 1.0))
    # This makes beta larger when sigma is large
    beta_base = 2.0
    sigma_mean = np.mean(sigma)
    # Scale beta based on typical sigma
    beta = beta_base * (sigma / (sigma + sigma_mean + 1e-9))
    ucb = _ucb_score(mu, sigma, beta)
    
    # Blend: 60% EI (cost-aware), 40% UCB
    # Normalize UCB to be comparable to EI
    # EI is typically in [0, f_best - min], UCB is in unbounded range
    # Let's normalize UCB by its max
    ucb_max = np.max(ucb) + 1e-9
    ucb_norm = ucb / ucb_max
    
    # EI is already cost-aware, UCB is not. Let's make UCB cost-aware too
    ucb_cost = ucb / (cost + 1e-9)
    ucb_cost_max = np.max(ucb_cost) + 1e-9
    ucb_cost_norm = ucb_cost / ucb_cost_max
    
    # Final score: blend of cost-aware EI and cost-aware UCB
    score = 0.6 * ei_cost + 0.4 * ucb_cost_norm
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
