import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s. Keep a safety margin.
_start_time = time.time()
_DEADLINE = _start_time + 320.0  # 10s safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (Three Hump Camel 2D).
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, the best (lowest) function value observed so far.
        cost: (2000,) array of evaluation costs for each candidate.
        
    Returns:
        (2000,) array of finite scores (higher = evaluate next).
    """
    # Deadline guard: if we're running out of time, return a fast fallback
    if time.time() > _DEADLINE:
        sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-9)
        return np.where(np.isfinite(ei / cost_safe), ei / cost_safe, 0.0)

    # Ensure inputs are numpy arrays of float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Handle small sigma separately to avoid numerical issues
    eps = 1e-8
    small_sigma = sigma < eps
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-8)
    
    scores = np.zeros_like(mu)
    
    # For small sigma, EI approximates max(0, f_best - mu)
    if np.any(small_sigma):
        improvement = f_best - mu[small_sigma]
        scores[small_sigma] = np.maximum(improvement, 0.0) / cost_safe[small_sigma]
    
    # For normal sigma, compute EI properly
    if np.any(~small_sigma):
        mu_valid = mu[~small_sigma]
        sigma_valid = sigma[~small_sigma]
        cost_valid = cost_safe[~small_sigma]
        
        # z = (f_best - mu) / sigma
        z = (f_best - mu_valid) / sigma_valid
        
        # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        
        ei = (f_best - mu_valid) * cdf_z + sigma_valid * pdf_z
        
        # Clamp to non-negative
        ei = np.maximum(ei, 0.0)
        
        scores[~small_sigma] = ei / cost_valid
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
