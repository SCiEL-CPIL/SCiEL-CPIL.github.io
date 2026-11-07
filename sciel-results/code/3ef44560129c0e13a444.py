"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline management
_START_TIME = time.monotonic()
_DEADLINE_SECONDS = 285.0  # Safety margin before 300s limit

# Track call count for adaptive behavior
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    global _call_count
    
    # Check deadline
    elapsed = time.monotonic() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        return np.zeros_like(mu, dtype=np.float64)
    
    # Increment call count
    _call_count += 1
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle any NaN or Inf values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is strictly positive for division
    sigma = np.maximum(sigma, 1e-12)
    
    # Ensure cost is within reasonable bounds for Rastrigin task
    # Cost is 1 + mean(x_scaled), so [1, 2]
    cost = np.clip(cost, 1.0, 2.0)
    
    n = len(mu)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Cost-effective EI: EI per unit cost
    cost_effective_ei = ei / cost
    
    # UCB component for exploration
    beta = 2.0
    ucb = mu + beta * sigma
    ucb_improvement = np.maximum(f_best - ucb, 0.0)
    ucb_cost = ucb_improvement / cost
    
    # Probability of Improvement
    pi = Phi_z
    pi_cost = pi / cost
    
    # Adaptive weighting based on call count and uncertainty
    # Early on, favor exploration (UCB); later, favor exploitation (EI)
    # Also consider the overall uncertainty level
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma) if n > 0 else 1.0
    uncertainty_ratio = sigma_mean / (sigma_max + 1e-9)
    
    # Decay exploration weight with call count
    # First few calls: more exploration, later: more exploitation
    if _call_count <= 3:
        ucb_weight = 0.5
        ei_weight = 0.4
        pi_weight = 0.1
    elif _call_count <= 8:
        ucb_weight = 0.35
        ei_weight = 0.55
        pi_weight = 0.1
    else:
        ucb_weight = 0.25
        ei_weight = 0.65
        pi_weight = 0.1
    
    # Normalize each component
    def normalize(arr):
        arr_max = np.max(arr)
        if arr_max > 1e-12:
            return arr / arr_max
        return np.zeros_like(arr)
    
    ce_norm = normalize(cost_effective_ei)
    ucb_norm = normalize(ucb_cost)
    pi_norm = normalize(pi_cost)
    
    # Combine with adaptive weights
    combined = ei_weight * ce_norm + ucb_weight * ucb_norm + pi_weight * pi_norm
    
    # Add exploration bonus for moderate uncertainty regions
    # This encourages exploring regions where we're not too certain but not totally unknown
    sigma_norm_local = sigma / (sigma_max + 1e-9)
    # Gaussian peak at moderate uncertainty (around 0.4-0.6)
    exploration_bonus = 0.1 * np.exp(-0.5 * ((sigma_norm_local - 0.5) / 0.25) ** 2)
    
    combined += exploration_bonus
    
    # Rank-based scoring for robustness
    # Higher rank = better score
    if n > 1:
        score_rank = np.argsort(np.argsort(combined)) / (n - 1)
    else:
        score_rank = np.ones_like(combined)
    
    # Cost-aware adjustment: slightly penalize expensive points
    # But not too aggressively, as we want to explore even if slightly more expensive
    cost_norm = (cost - np.min(cost)) / (np.max(cost) - np.min(cost) + 1e-9)
    
    # Final score: blend rank-based score with a small cost penalty
    final_score = 0.85 * score_rank + 0.15 * (1.0 - cost_norm)
    
    # Ensure finite output
    final_score = np.nan_to_num(final_score, nan=0.0, posinf=1.0, neginf=0.0)
    
    return final_score
# EVOLVE-BLOCK-END
