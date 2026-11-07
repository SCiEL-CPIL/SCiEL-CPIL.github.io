"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 295.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're getting close to the 300s limit, return safe scores
    if time.time() - _start_time > _TIMEOUT:
        return np.ones_like(mu)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf and norm.pdf
    z = np.clip(z, -10, 10)
    
    # Standard Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle any remaining numerical issues
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Cost-aware: divide by cost to account for expensive evaluations
    cost_safe = np.maximum(cost, 1e-9)
    
    # CEPI: Cost-Effective Probability of Improvement
    # Use a dynamically relaxed threshold to prioritize meaningful gains
    # The threshold is f_best - 0.05, so we want to beat this value
    # PI with relaxed threshold: Phi((threshold - mu) / sigma) where threshold = f_best - 0.05
    # But for minimization, we want mu < threshold, so z_relaxed = (f_best - 0.05 - mu) / sigma
    # Actually, for minimization, improvement means mu < f_best. 
    # Let's use the standard PI but with a relaxed target:
    # We want P(f(x) < f_best - 0.05) = Phi((f_best - 0.05 - mu) / sigma)
    # This is more selective - only points with high confidence of beating f_best by at least 0.05
    
    # Compute relaxed z-score for CEPI
    relaxed_target = f_best - 0.05
    z_relaxed = (relaxed_target - mu) / sigma_safe
    z_relaxed = np.clip(z_relaxed, -10, 10)
    pi_relaxed = norm.cdf(z_relaxed)
    pi_relaxed = np.nan_to_num(pi_relaxed, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Cost-aware CEPI
    cepi_score = pi_relaxed / cost_safe
    
    # UCB term for exploration
    beta = 2.0
    ucb = -mu + beta * sigma_safe
    ucb_score = ucb / cost_safe
    
    # EI score
    ei_score = ei / cost_safe
    
    # Combine: primarily CEPI (exploitative, cost-aware, selective) 
    # with some UCB (exploratory) and EI (average improvement)
    # Weighting: CEPI gets the highest weight as it's the most selective
    score = 0.5 * cepi_score + 0.3 * ucb_score + 0.2 * ei_score
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
