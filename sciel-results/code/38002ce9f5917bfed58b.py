"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _expected_improvement(mu, sigma, f_best):
    """Standard EI for minimization."""
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For numerical stability, handle large z
    ei = np.zeros_like(mu)
    # Case 1: f_best - mu <= 0 (mu >= f_best), EI is 0
    mask_pos = (f_best - mu) > 0
    if np.any(mask_pos):
        z_pos = z[mask_pos]
        mu_pos = mu[mask_pos]
        sigma_pos = sigma_safe[mask_pos]
        ei_pos = (f_best - mu_pos) * norm.cdf(z_pos) + sigma_pos * norm.pdf(z_pos)
        # For very large z, cdf -> 1, pdf -> 0, so EI -> f_best - mu
        # For very small z, this formula is fine
        ei[mask_pos] = ei_pos
    return ei


def _upper_confidence_bound(mu, sigma, f_best, beta):
    """UCB for minimization: lower is better, so we want to minimize mu - beta*sigma"""
    return mu - beta * sigma


def _probability_of_improvement(mu, sigma, f_best):
    """POI for minimization."""
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    return norm.cdf(z)


def _qei(mu, sigma, f_best, xi):
    """Expected Improvement with exploration parameter xi."""
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu - xi) / sigma_safe
    ei = np.zeros_like(mu)
    mask_pos = (f_best - mu - xi) > 0
    if np.any(mask_pos):
        z_pos = z[mask_pos]
        mu_pos = mu[mask_pos]
        sigma_pos = sigma_safe[mask_pos]
        ei_pos = (f_best - mu_pos - xi) * norm.cdf(z_pos) + sigma_pos * norm.pdf(z_pos)
        ei[mask_pos] = ei_pos
    return ei


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    mu, sigma: arrays of shape (2000,) - GP mean and std in objective units
    f_best: scalar - best observed value so far
    cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns: array of shape (2000,) with finite scores (higher = evaluate next)
    """
    # Deadline guard
    start_time = time.time()
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-6)
    
    # Compute multiple acquisition scores and combine them
    # 1. EI per unit cost
    ei = _expected_improvement(mu, sigma, f_best)
    ei_cost = ei / cost
    
    # 2. QEI with small exploration per unit cost
    qei = _qei(mu, sigma, f_best, xi=0.01)
    qei_cost = qei / cost
    
    # 3. UCB per unit cost (for minimization, lower UCB is better, so we negate)
    beta = 2.0
    ucb = _upper_confidence_bound(mu, sigma, f_best, beta)
    ucb_cost = -ucb / cost
    
    # 4. POI per unit cost
    poi = _probability_of_improvement(mu, sigma, f_best)
    poi_cost = poi / cost
    
    # Normalize each score to [0, 1] range for fair combination
    def normalize(arr):
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        if arr_max - arr_min < 1e-12:
            return np.zeros_like(arr)
        return (arr - arr_min) / (arr_max - arr_min)
    
    ei_norm = normalize(ei_cost)
    qei_norm = normalize(qei_cost)
    ucb_norm = normalize(ucb_cost)
    poi_norm = normalize(poi_cost)
    
    # Weighted combination
    # EI is generally most robust, give it highest weight
    # QEI adds slight exploration
    # UCB provides exploration
    # POI is a simpler heuristic
    weights = np.array([0.4, 0.25, 0.2, 0.15])
    
    combined = (weights[0] * ei_norm + 
                weights[1] * qei_norm + 
                weights[2] * ucb_norm + 
                weights[3] * poi_norm)
    
    # Ensure all values are finite
    combined = np.where(np.isfinite(combined), combined, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 290:  # Safety margin before 300s limit
        return combined
    
    return combined
# EVOLVE-BLOCK-END
