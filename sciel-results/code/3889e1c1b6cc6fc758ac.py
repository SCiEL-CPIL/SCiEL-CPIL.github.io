import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check if we're approaching the deadline (330s limit, use 325s as safety margin)
    elapsed = time.time() - _start_time
    if elapsed > 325:
        safe_cost = np.maximum(cost, 1e-10)
        return -mu / safe_cost
    
    # Small epsilon to prevent division by zero
    eps = 1e-12
    safe_sigma = np.maximum(sigma, eps)
    safe_cost = np.maximum(cost, eps)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / safe_sigma
    z_clipped = np.clip(z, -10, 10)
    
    cdf_val = norm.cdf(z_clipped)
    pdf_val = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * cdf_val + safe_sigma * pdf_val
    
    # Cost-aware EI
    c_ei = ei / safe_cost
    
    # Adaptive exploration: scale with how far we are from the optimum
    # When f_best is large (far from 0), explore more; when close, exploit
    # Griewank minimum is 0, so f_best > 0 always
    # Scale exploration inversely with f_best, with a floor
    exploration_scale = 0.05 / (abs(f_best) + 0.01)
    exploration_scale = np.clip(exploration_scale, 0.01, 0.15)
    
    # Exploration bonus proportional to uncertainty per cost
    exploration_bonus = exploration_scale * safe_sigma / safe_cost
    
    # Light UCB component for early exploration
    # UCB = -mu + beta * sigma, where beta adapts to uncertainty spread
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    if sigma_mean > eps:
        beta = 1.0 + 0.3 * (sigma_std / (sigma_mean + eps))
    else:
        beta = 1.0
    beta = np.clip(beta, 0.5, 2.0)
    
    ucb = -mu + beta * safe_sigma
    c_ucb = ucb / safe_cost
    
    # Blend EI and UCB: weight EI more heavily as we converge
    # When f_best is small (close to optimum), rely more on EI
    ei_weight = 0.85
    ucb_weight = 0.15
    
    score = ei_weight * c_ei + ucb_weight * c_ucb + exploration_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
