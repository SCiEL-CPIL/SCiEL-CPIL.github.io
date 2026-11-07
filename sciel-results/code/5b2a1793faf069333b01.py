"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm


def _compute_ei(mu, sigma, f_best):
    """Expected Improvement."""
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return ei


def _compute_ucb(mu, sigma, beta):
    """Upper Confidence Bound (for minimization, we use lower bound)."""
    return mu - beta * sigma


def _compute_pucb(mu, sigma, beta):
    """Probability of Improvement weighted by UCB-like term."""
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best_val - mu) / sigma_safe
    pi = norm.cdf(z)
    return pi * (1 + beta * sigma_safe)


def acquisition(mu, sigma, f_best, cost):
    global _deadline
    if _deadline is None:
        _deadline = time.time() + 280  # 280s safety margin before 300s limit
    
    n = len(mu)
    
    # Check if we're close to deadline
    elapsed = time.time()
    if elapsed > _deadline - 5:
        # Emergency: return simple EI / cost
        eps = 1e-9
        sigma_safe = np.maximum(sigma, eps)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.maximum(ei / np.maximum(cost, 1e-6), 1e-10)
    
    # Cost normalization: cost is in [1, 2] for x in [0,1]^6
    cost_safe = np.maximum(cost, 1e-6)
    
    # Compute EI
    ei = _compute_ei(mu, sigma, f_best)
    
    # Compute UCB with adaptive beta
    # Use a beta that balances exploration
    beta = 0.5
    
    # Compute UCB (lower confidence bound for minimization)
    ucb = _compute_ucb(mu, sigma, beta)
    
    # Normalize EI and UCB to comparable scales
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-9
    ei_norm = (ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb) + 1e-9
    ucb_norm = (ucb - ucb_mean) / ucb_std
    
    # Blend EI and UCB
    # Weight: more UCB early (high uncertainty), more EI later (converging)
    # Use sigma as a proxy for uncertainty level
    mean_sigma = np.mean(sigma)
    # If mean sigma is high, weight UCB more
    ucb_weight = np.clip(0.3 + 0.4 * mean_sigma, 0.3, 0.7)
    ei_weight = 1.0 - ucb_weight
    
    blended = ei_weight * ei_norm + ucb_weight * ucb_norm
    
    # Cost-aware: divide by cost
    scores = blended / cost_safe
    
    # Ensure finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores


# Deadline guard
_deadline = None
# EVOLVE-BLOCK-END
