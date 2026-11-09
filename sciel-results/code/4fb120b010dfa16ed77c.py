import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import time

# Deadline guard: 300s per call, keep safety margin
_DEADLINE_SEC = 290.0

def _rastrigin_2d(x):
    """Rastrigin 2D on [-5.12, 5.12]^2, min 0 at (0,0)."""
    A = 10.0
    a = x[0] ** 2 + x[1] ** 2
    b = 2.0 * A * (np.cos(2.0 * np.pi * x[0]) + np.cos(2.0 * np.pi * x[1]))
    return A * 2.0 + a + b

def _scale_to_unit(x, lo=-5.12, hi=5.12):
    """Scale from [-5.12, 5.12] to [0, 1]."""
    return (x - lo) / (hi - lo)

def _scale_from_unit(u, lo=-5.12, hi=5.12):
    """Scale from [0, 1] back to [-5.12, 5.12]."""
    return lo + u * (hi - hi) + u * (hi - lo)

def _expected_improvement_cost(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: EI per unit cost, plus exploration bonus.
    mu, sigma, f_best, cost are 1-D arrays of shape (N,).
    Returns N finite scores (higher = more promising).
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip negative EI to 0 (we only care about improvement)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to reward cheap evaluations
    cost_safe = np.maximum(cost, 1e-9)
    score = ei / cost_safe
    
    # Add small exploration bonus for high uncertainty to avoid exploitation trap
    # This helps when GP hasn't explored enough
    exploration_bonus = sigma_safe / cost_safe * 0.05
    
    total_score = score + exploration_bonus
    
    # Ensure finiteness
    total_score = np.where(np.isfinite(total_score), total_score, 0.0)
    
    return total_score

def _ucb_cost(mu, sigma, f_best, cost, kappa=1.0):
    """
    Cost-aware UCB: mu + kappa * sigma, divided by cost.
    Higher means more promising (since we're minimizing, we want low mu + high sigma).
    Actually for minimization: score = -(mu - kappa*sigma)/cost = (kappa*sigma - mu)/cost
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # For minimization, we want to maximize: -mu + kappa*sigma
    # Then divide by cost for cost-awareness
    score = (kappa * sigma_safe - mu) / cost_safe
    
    score = np.where(np.isfinite(score), score, 0.0)
    return score

def _pi_cost(mu, sigma, f_best, cost, xi=0.01):
    """
    Probability of improvement per unit cost.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    z = (f_best - mu - xi) / sigma_safe
    pi = norm.cdf(z)
    
    score = pi / cost_safe
    score = np.where(np.isfinite(score), score, 0.0)
    return score

def _hybrid_acquisition(mu, sigma, f_best, cost):
    """
    Combine multiple acquisition functions with adaptive weights.
    """
    # EI-based (exploitation)
    ei_score = _expected_improvement_cost(mu, sigma, f_best, cost)
    
    # UCB-based (exploration)
    ucb_score = _ucb_cost(mu, sigma, f_best, cost, kappa=1.0)
    
    # Normalize each to [0, 1] range for combining
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    ei_norm = normalize(ei_score)
    ucb_norm = normalize(ucb_score)
    
    # Weight: more EI when we have good uncertainty estimates, more UCB early
    # Use ratio of sigma to f_best magnitude as heuristic
    sigma_mean = np.mean(sigma)
    f_best_abs = abs(f_best)
    
    # When f_best is close to 0 (good), rely more on EI
    # When f_best is large (bad), rely more on exploration
    if f_best_abs < 1.0:
        w_ei = 0.8
        w_ucb = 0.2
    elif f_best_abs < 10.0:
        w_ei = 0.5
        w_ucb = 0.5
    else:
        w_ei = 0.3
        w_ucb = 0.7
    
    combined = w_ei * ei_norm + w_ucb * ucb_norm
    combined = np.where(np.isfinite(combined), combined, 0.0)
    
    return combined

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Rastrigin 2D.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions  
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    - array of shape (2000,) with finite scores (higher = evaluate next)
    """
    t_start = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Check deadline
    if time.time() - t_start > _DEADLINE_SEC:
        return np.zeros(n)
    
    # Compute hybrid acquisition score
    scores = _hybrid_acquisition(mu, sigma, f_best, cost)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final deadline check
    if time.time() - t_start > _DEADLINE_SEC:
        # Return what we have
        return scores
    
    return scores
