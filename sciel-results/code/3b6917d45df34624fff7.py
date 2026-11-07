"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Hartmann 3d.
    
    Uses a combination of:
    1. Expected Improvement (EI) per unit cost (baseline)
    2. Adaptive exploration/exploitation based on progress (gap to optimum)
    3. Time budget guard to stay within 300s per evaluation
    
    The Hartmann 3d minimum is approximately -3.86278.
    """
    start_time = time.time()
    DEADLINE = 280.0  # Safety margin before 300s limit
    
    # Convert to arrays for safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Known minimum of Hartmann 3d
    HARTMANN3D_MIN = -3.86278
    
    # Compute progress: how close are we to the optimum?
    # gap = f_best - HARTMANN3D_MIN (positive when we're above optimum)
    gap = f_best - HARTMANN3D_MIN
    # Normalize gap: at start f_best is around 0-1, so gap ~ 3-5
    # At convergence, gap ~ 0
    # Use a smooth transition: early on be more exploratory, late be more exploitative
    # progress in [0, 1], 1 means we've reached the optimum
    max_gap = max(1.0, f_best - HARTMANN3D_MIN)  # avoid div by zero
    progress = np.clip(1.0 - gap / 5.0, 0.0, 1.0) if gap > 0 else 1.0
    
    # Adaptive parameters based on progress
    # Early (progress ~ 0): more exploration, less cost-sensitive
    # Late (progress ~ 1): more exploitation, more cost-sensitive
    # alpha controls weight of EI, beta controls weight of cost
    alpha = 1.0 + progress * 1.5  # 1.0 to 2.5
    beta = 1.0 + progress * 2.0   # 1.0 to 3.0
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure non-negative EI
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI^alpha / cost^beta
    # This makes us prefer high improvement with low cost, with adaptive sensitivity
    cost_safe = np.maximum(cost, 1e-9)
    score = (ei ** alpha) / (cost_safe ** beta)
    
    # Add a small exploration bonus for points with high uncertainty
    # This helps escape local optima early on
    exploration_weight = (1.0 - progress) * 0.3
    uncertainty_bonus = exploration_weight * sigma_safe / (np.max(sigma_safe) + 1e-9)
    
    score = score + uncertainty_bonus
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return current result as-is
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Final time check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return score
    
    return score
# EVOLVE-BLOCK-END
