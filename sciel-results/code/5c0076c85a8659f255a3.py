"""Rewritten program with a stronger cost-aware acquisition for rastrigin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _expected_improvement(mu, sigma, f_best, xi=0.0):
    """Expected improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-9)
    improvement = f_best - mu - xi
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return ei


def _ucb(mu, sigma, beta):
    """Upper confidence bound for minimization (lower is better, so we negate)."""
    # For minimization, we want to explore where mu is low and sigma is high
    # UCB for minimization: mu - beta * sigma (lower value = more promising)
    # But since we want higher score = more promising, we use: -(mu - beta*sigma) = -mu + beta*sigma
    return -mu + beta * sigma


def _pi(mu, sigma, f_best):
    """Probability of improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    return norm.cdf(z)


def _kullback_leibler(mu, sigma, f_best):
    """Kullback-Leibler divergence based acquisition."""
    sigma_safe = np.maximum(sigma, 1e-9)
    # KL divergence between N(mu, sigma^2) and N(f_best, sigma^2)
    # This rewards points where the GP is uncertain about whether the point is better than f_best
    kl = 0.5 * ((f_best - mu) ** 2) / (sigma_safe ** 2)
    return kl


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing rastrigin 2d.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Check time
    if time.time() > deadline:
        # Fallback: just return cost-normalized EI
        ei = _expected_improvement(mu, sigma, f_best)
        scores = ei / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Compute multiple acquisition components
    ei = _expected_improvement(mu, sigma, f_best)
    pi = _pi(mu, sigma, f_best)
    kl = _kullback_leibler(mu, sigma, f_best)
    
    # Adaptive beta for UCB: balance exploration and exploitation
    # Use a beta that adapts based on the spread of sigma
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    # Higher beta when there's more uncertainty overall
    beta = 1.0 + 0.5 * (sigma_std / (sigma_mean + 1e-9))
    beta = np.clip(beta, 0.5, 3.0)
    ucb = _ucb(mu, sigma, beta)
    
    # Normalize each component to [0, 1] range for better combination
    def _normalize(arr):
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        arr_range = arr_max - arr_min
        if arr_range < 1e-9:
            return np.zeros_like(arr)
        return (arr - arr_min) / arr_range
    
    ei_norm = _normalize(ei)
    pi_norm = _normalize(pi)
    ucb_norm = _normalize(ucb)
    kl_norm = _normalize(kl)
    
    # Check time
    if time.time() > deadline:
        scores = (ei_norm + ucb_norm) / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Weighted combination of acquisition components
    # EI: exploitation (find points likely to be better than f_best)
    # UCB: exploration (find points with high uncertainty)
    # PI: moderate exploitation
    # KL: exploration (find points where GP is uncertain)
    w_ei = 0.4
    w_ucb = 0.3
    w_pi = 0.1
    w_kl = 0.2
    
    combined = w_ei * ei_norm + w_ucb * ucb_norm + w_pi * pi_norm + w_kl * kl_norm
    
    # Cost-aware normalization: divide by cost to prefer cheaper evaluations
    # Use a power to control the strength of cost preference
    cost_power = 0.5
    cost_factor = cost_safe ** cost_power
    scores = combined / cost_factor
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
