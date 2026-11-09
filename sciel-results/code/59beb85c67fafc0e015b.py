"""Rewritten program using a hybrid UCB/EI acquisition with adaptive exploration."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _remaining():
    # Leave a safety margin of 2.0s
    return max(0.0, 328.0 - _elapsed())

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses a hybrid approach combining UCB and Expected Improvement,
    with adaptive exploration strength based on the spread of predictions.
    Returns 2000 finite scores; higher means evaluate next.
    """
    if _elapsed() > 328.0:
        return np.zeros_like(mu)
    
    # Ensure inputs are proper arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases - ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-12)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Compute Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    z = np.clip(z, -10.0, 10.0)
    
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB (Upper Confidence Bound) - for minimization, we use lower bound
    # UCB = mu - beta * sigma (we want to explore where mu is low or sigma is high)
    # The beta parameter controls exploration-exploitation tradeoff
    
    # Adaptive beta: start with higher exploration, decrease over time
    # Use the spread of mu and sigma to adapt
    mu_std = np.std(mu)
    sigma_mean = np.mean(sigma_safe)
    
    # Base beta that encourages exploration
    beta_base = 2.0
    
    # Adapt based on the relative scale of uncertainty
    if mu_std > 1e-6:
        # If predictions have significant spread, reduce exploration slightly
        # as we're getting better estimates
        beta = beta_base * (1.0 + 0.5 * (sigma_mean / (mu_std + 1e-10)))
    else:
        beta = beta_base
    
    # Clip beta to reasonable range
    beta = np.clip(beta, 0.5, 4.0)
    
    # UCB score (higher is better for our purposes - we want low mu and high sigma)
    ucb = mu - beta * sigma_safe
    
    # Normalize both EI and UCB to [0, 1] range for fair combination
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.ones_like(ei)
    
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        # For UCB in minimization, lower is better, so we invert
        ucb_norm = (ucb_max - ucb) / ucb_range
    else:
        ucb_norm = np.ones_like(ucb)
    
    # Combine EI and UCB with adaptive weights
    # Weight based on how much uncertainty remains
    sigma_relative = sigma_mean / (np.abs(f_best) + 1e-10)
    sigma_relative = np.clip(sigma_relative, 0.0, 1.0)
    
    # When uncertainty is high, weight UCB more; when low, weight EI more
    w_ucb = 0.3 + 0.4 * sigma_relative
    w_ei = 1.0 - w_ucb
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Add a small pure exploration bonus for very high uncertainty regions
    # This helps escape local minima in the multimodal Michalewicz function
    high_uncertainty_mask = sigma_safe > (np.percentile(sigma_safe, 90) + 1e-10)
    exploration_bonus = np.where(high_uncertainty_mask, 0.02 * sigma_safe, 0.0)
    
    scores = scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
