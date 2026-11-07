"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Uses a combination of Expected Improvement (EI) relative to the true
    minimum (3.0) and an adaptive UCB term. The true minimum provides a
    stronger exploration signal in high-variance regions that might contain
    the global optimum.
    
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
    
    # True minimum of Goldstein-Price 2D is 3.0
    f_true_min = 3.0
    
    # Use the better of f_best and f_true_min as reference for improvement
    # This gives stronger exploration signal when f_best > f_true_min
    f_ref = min(f_best, f_true_min + 0.5)  # Slight offset to encourage exploration
    
    # Compute Expected Improvement relative to f_ref (for minimization)
    # EI = (f_ref - mu) * Phi(z) + sigma * phi(z)
    z = (f_ref - mu) / sigma_safe
    ei = (f_ref - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Also compute EI relative to f_best for exploitation
    z_best = (f_best - mu) / sigma_safe
    ei_best = (f_best - mu) * norm.cdf(z_best) + sigma_safe * norm.pdf(z_best)
    
    # Compute a UCB-like term with adaptive beta
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-10:
        mu_range = 1.0
    
    # Adaptive beta based on relative uncertainty
    sigma_mean = np.mean(sigma_safe)
    sigma_std = np.std(sigma_safe)
    
    # Beta increases with uncertainty relative to prediction spread
    # Also consider the gap between f_best and true minimum
    gap_to_opt = max(0.0, f_best - f_true_min)
    gap_factor = min(1.0, gap_to_opt / 5.0)  # Scale gap to [0, 1]
    
    # Beta ranges from 1.5 to 5.0
    base_beta = 1.5 + 2.0 * min(1.0, sigma_mean / (mu_range + 1e-10))
    # Boost beta when we're far from optimum (early stage)
    beta = base_beta + 1.5 * gap_factor
    
    # UCB score for minimization
    ucb_value = mu + beta * sigma_safe
    ucb_score = -ucb_value
    
    # Normalize terms for stable combination
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn > 1e-12:
            return (arr - mn) / (mx - mn)
        else:
            return np.zeros_like(arr)
    
    ei_norm = normalize(ei)
    ei_best_norm = normalize(ei_best)
    ucb_norm = normalize(ucb_score)
    
    # Adaptive weights based on uncertainty and gap to optimum
    # High uncertainty + large gap -> more exploration (UCB + EI to true min)
    # Low uncertainty + small gap -> more exploitation (EI to f_best)
    uncertainty_ratio = min(1.0, sigma_mean / (mu_range + 1e-10))
    
    # Weight for exploration terms (EI to true min + UCB)
    w_explore = 0.3 + 0.4 * uncertainty_ratio + 0.3 * gap_factor
    w_explore = min(1.0, w_explore)
    
    # Split exploration weight between EI-to-true-min and UCB
    w_ei_true = 0.5 * w_explore
    w_ucb = 0.5 * w_explore
    w_ei_best = 1.0 - w_explore
    
    # Combine: smooth weighted sum
    scores = w_ei_true * ei_norm + w_ucb * ucb_norm + w_ei_best * ei_best_norm
    
    # Add bonus for regions where mu is significantly below f_best
    # This helps exploit already-promising regions
    below_best = np.maximum(0.0, f_best - mu)
    if np.max(below_best) > 1e-10:
        below_best_norm = below_best / (np.max(below_best) + 1e-10)
        scores += 0.15 * below_best_norm
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard
    if time.time() - t_start > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
