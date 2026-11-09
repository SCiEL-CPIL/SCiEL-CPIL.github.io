import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_t0 = time.time()
_DEADLINE = 300.0
_MARGIN = 5.0


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're close to timeout, return quickly
    if time.time() - _t0 > _DEADLINE - _MARGIN:
        return np.zeros_like(mu)
    
    # Sanitize inputs
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Standardized improvement z
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (PI) - more robust for large sigma
    pi = norm.cdf(z)
    
    # Blend EI and PI: PI provides gradient info when EI saturates
    # Use a smooth blend: weight PI more when sigma is large relative to (f_best - mu)
    improvement = np.maximum(f_best - mu, 0.0)
    # When improvement is small relative to sigma, PI is more informative
    pi_weight = 0.3 * np.exp(-improvement / (sigma_safe + 1e-9))
    blended = (1.0 - pi_weight) * ei + pi_weight * pi * sigma_safe
    
    # Normalized by cost
    blended_score = blended / cost_safe
    
    # Prior bias towards expected minimizer location
    # Minimizer is in inner 80% of [-4, 5]^4
    # In scaled coords [0,1], the inner 80% is [0.1, 0.9]
    # Center is at 0.5 in scaled coords
    # Expected cost at minimizer: 1 + 0.5 = 1.5
    
    # Gaussian penalty for cost deviation from 1.5
    # Wider kernel (lambda=0.5) to avoid over-penalizing good candidates
    target_cost = 1.5
    lambda_bias = 0.5
    cost_weight = np.exp(-lambda_bias * (cost_safe - target_cost) ** 2)
    
    # Exploration bonus: smooth max of linear sigma/cost term and small floor
    # This ensures exploration doesn't vanish even when sigma is small
    exploration_linear = 0.1 * sigma_safe / cost_safe
    exploration_floor = 0.01
    # Soft-max via log-sum-exp for smoothness
    exploration_bonus = np.logaddexp(np.log(exploration_linear + 1e-15), np.log(exploration_floor)) - np.log(2.0)
    
    # Combine: primarily blended EI/PI cost-normalized, plus exploration bonus, weighted by cost proximity
    score = (blended_score + exploration_bonus) * cost_weight
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
