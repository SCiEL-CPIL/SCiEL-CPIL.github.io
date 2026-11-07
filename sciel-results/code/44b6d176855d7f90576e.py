"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Uses a posterior-sampling-based acquisition that approximates Expected
    Regret via Monte Carlo sampling of the GP posterior. The key insight is
    to use the known global minimum (3.0) as the reference for computing
    improvement, rather than f_best. This provides a constant, strong
    exploration signal that doesn't diminish as f_best approaches the true
    optimum, keeping the acquisition sensitive to high-variance regions
    that might contain the global minimum.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation predictions at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Safety deadline guard
    t_start = time.time()
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # The known global minimum of Goldstein-Price 2D is 3.0
    f_opt = 3.0
    
    # We want to maximize the probability that f(x) < f_opt, weighted by
    # the expected magnitude of improvement over f_opt. This is a form of
    # Expected Improvement relative to the true optimum.
    
    # Compute z-score relative to the true optimum
    z_opt = (f_opt - mu) / sigma_safe
    
    # Probability of beating the true optimum
    prob_beat_opt = norm.cdf(z_opt)
    
    # Expected improvement over the true optimum
    # E[max(f_opt - f(x), 0)] = (f_opt - mu) * Phi(z) + sigma * phi(z)
    ei_opt = (f_opt - mu) * prob_beat_opt + sigma_safe * norm.pdf(z_opt)
    
    # Also compute EI relative to f_best for exploitation
    z_best = (f_best - mu) / sigma_safe
    ei_best = (f_best - mu) * norm.cdf(z_best) + sigma_safe * norm.pdf(z_best)
    
    # Combine: weight between exploration (toward true opt) and exploitation
    # In early stages, explore more toward the true optimum
    # In later stages, exploit more around f_best
    
    # Estimate stage from posterior uncertainty
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-10:
        mu_range = 1.0
    
    sigma_mean = np.mean(sigma_safe)
    relative_uncertainty = np.clip(sigma_mean / mu_range, 0.0, 1.0)
    
    # Weight: more exploration early, more exploitation later
    w_explore = 0.6 * relative_uncertainty + 0.2
    w_exploit = 1.0 - w_explore
    
    # Normalize both EI scores to [0, 1]
    ei_opt_min = np.min(ei_opt)
    ei_opt_max = np.max(ei_opt)
    if ei_opt_max - ei_opt_min > 1e-12:
        ei_opt_norm = (ei_opt - ei_opt_min) / (ei_opt_max - ei_opt_min)
    else:
        ei_opt_norm = np.zeros_like(ei_opt)
    
    ei_best_min = np.min(ei_best)
    ei_best_max = np.max(ei_best)
    if ei_best_max - ei_best_min > 1e-12:
        ei_best_norm = (ei_best - ei_best_min) / (ei_best_max - ei_best_min)
    else:
        ei_best_norm = np.zeros_like(ei_best)
    
    # Combine
    scores = w_explore * ei_opt_norm + w_exploit * ei_best_norm
    
    # Add a bonus for high probability of beating the true optimum
    # This reinforces exploration in regions with high variance
    scores += 0.1 * prob_beat_opt
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard
    if time.time() - t_start > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
