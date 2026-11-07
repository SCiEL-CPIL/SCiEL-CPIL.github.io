"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global timing guard
_START_TIME = time.time()
TIME_LIMIT = 320.0  # Leave 10s safety margin for 330s limit

def _time_ok():
    return (time.time() - _START_TIME) < TIME_LIMIT

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Styblinski-Tang 2D minimization.
    
    Uses a hybrid approach:
    1. Expected Improvement (EI) for exploitation
    2. Knowledge-based prior for Styblinski-Tang global minimum region
    3. Cost-aware weighting to prefer cheaper evaluations
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    if not _time_ok():
        # Return simple cost-based scores to finish quickly
        return -cost
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Knowledge-based prior for Styblinski-Tang
    # The global minimum of ST2D is at x* = (-2.903534, -2.903534) with f(x*) = -78.3323
    # But the problem is shifted: f_k(x) = f(x - s_k) where s_k moves the minimizer
    # to a uniform point in the inner 80% of each range
    # Inner 80% of [-5, 5] is [-3, 3], so minimizer is in [-3, 3]^2
    
    # We don't know s_k directly, but we can use the GP to infer the minimum region
    # Structure-aware prior: combine GP mean-based minimum detection with cost weighting
    
    # Infer approximate minimum location from GP mean
    # Find the candidate with lowest GP mean
    mu_min_idx = np.argmin(mu)
    mu_min_val = mu[mu_min_idx]
    
    # The GP mean should be low near the true minimum
    # Create a "minimum region" indicator based on how close mu is to mu_min
    # Use a smooth kernel around the GP-predicted minimum region
    
    # Scale: typical improvement scale
    improvement_scale = max(np.std(mu), 1e-6)
    
    # Minimum-region bonus: high where GP mean is low
    # Use a Gaussian-like kernel centered on the GP minimum region
    # The "distance" in function value space
    min_region_bonus = np.exp(-((mu - mu_min_val) / improvement_scale) ** 2)
    
    # Early exploration bonus: when GP is uncertain, explore more
    # When GP is confident, exploit more
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    # Adaptive weight for exploration vs exploitation
    # Early on (high sigma), favor exploration; later (low sigma), favor exploitation
    exploration_weight = np.clip(sigma_mean / (sigma_max + 1e-9), 0.3, 0.7)
    
    # Cost normalization
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range < 1e-9:
        cost_norm = np.ones_like(cost) * 0.5
    else:
        cost_norm = (cost - cost_min) / cost_range
    
    # Cost penalty: prefer cheaper evaluations
    # Soft penalty that's not too aggressive
    cost_penalty = 1.0 / (1.0 + 3.0 * cost_norm)
    
    # Combine components
    # 1. EI component (exploitation)
    # 2. Minimum-region bonus (knowledge-based)
    # 3. Uncertainty bonus (exploration)
    
    # Scale EI to be comparable
    ei_scale = max(np.abs(f_best), 1.0)
    ei_scaled = ei / ei_scale
    
    # Uncertainty bonus: prefer high uncertainty regions
    uncertainty_bonus = sigma_safe / (sigma_max + 1e-9)
    
    # Weighted combination
    w_ei = 0.4 * (1.0 - exploration_weight)
    w_min_region = 0.3
    w_uncertainty = 0.3 * exploration_weight
    
    scores = (w_ei * ei_scaled + 
              w_min_region * min_region_bonus + 
              w_uncertainty * uncertainty_bonus)
    
    # Apply cost penalty
    scores = scores * cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
