"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Refined Cost-Aware Probability of Improvement (PI).
    
    The base strategy is PI, which scored 12.19. This refines it by introducing
    a soft cost penalty to the acquisition score. The Rosenbrock function minimum
    is located at the origin, which corresponds to the lowest possible evaluation cost
    (cost = 1 + mean(x_scaled)). High-cost points are further from the origin and thus
    less likely to contain the global minimum.
    
    By subtracting a small fraction of the normalized cost from the PI score, we
    bias the selection towards cheaper candidates when the probability of improvement
    is comparable. This prevents the optimizer from wasting budget on expensive regions
    that are only marginally more promising than cheaper alternatives.
    
    Formula:
    1. z = (f_best - mu) / sigma
    2. PI = Phi(z)
    3. cost_norm = (cost - min_cost) / (max_cost - min_cost)  # in [0, 1]
    4. score = PI - lambda * cost_norm
    
    lambda is set to 0.1 to ensure the PI term remains dominant while providing a
    meaningful tie-breaker in favor of low-cost evaluations.
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Safety: handle NaN/Inf
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute Probability of Improvement (PI)
    z = (f_best - mu) / sigma
    pi_score = norm.cdf(z)
    
    # Normalize cost to [0, 1] for penalty calculation
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    
    if cost_range > 1e-9:
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.zeros_like(cost)
        
    # Cost penalty weight
    # A small weight ensures PI is the primary driver, but cost breaks ties.
    lambda_cost = 0.1
    
    # Final score: Higher is better
    score = pi_score - lambda_cost * cost_norm
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin
         # Fallback to simple exploitation: -mu, adjusted slightly for cost if needed
         # But simple -mu is safer for validity
         score = -mu
         # Re-normalize cost penalty for fallback if we want consistency, 
         # but -mu is standard. Let's stick to -mu for safety.
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.5, posinf=1.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
