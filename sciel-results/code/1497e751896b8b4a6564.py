"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're close to the 330s limit, return a safe finite array
    # We don't track state here, but we can add a simple safeguard
    # Actually, the function is called many times, so we need to be fast.
    # Let's implement a hybrid acquisition function.
    
    # Handle edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    cost = np.maximum(cost, 1e-10)
    
    # Compute z-scores
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Adaptive weighting:
    # When f_best is large (early in search), favor exploration (PI)
    # When f_best is small (close to optimum), favor exploitation (EI)
    # Rastrigin minimum is 0, so f_best approaches 0 as we converge
    # Use a smooth transition based on f_best
    
    # Scale factor: when f_best is large, alpha is closer to 1 (more PI)
    # When f_best is small, alpha is closer to 0 (more EI)
    # Use a sigmoid-like transition
    # f_best for Rastrigin can be up to ~20 in 2D, typically starts around 10-15
    # Let's use a transition centered around f_best = 1.0
    
    alpha = 1.0 / (1.0 + np.exp((f_best - 1.0) * 2.0))
    # alpha is in [0, 1], closer to 1 when f_best < 1, closer to 0 when f_best > 1
    
    # Actually, we want:
    # - Early (f_best large): more PI (exploration)
    # - Late (f_best small): more EI (exploitation)
    # So alpha should be: weight on EI
    # Let's redefine:
    # weight_ei = sigmoid((f_best - threshold) * scale) -> 0 when f_best small, 1 when f_best large? No.
    # We want: f_best small -> high EI weight, f_best large -> high PI weight
    # So: weight_ei = 1 - sigmoid((f_best - threshold) * scale)
    # Or: weight_pi = sigmoid((f_best - threshold) * scale)
    
    threshold = 1.0
    scale = 2.0
    weight_pi = 1.0 / (1.0 + np.exp(-(f_best - threshold) * scale))
    weight_ei = 1.0 - weight_pi
    
    # Combine
    combined = weight_ei * ei + weight_pi * pi
    
    # Cost-aware: divide by cost
    result = combined / cost
    
    # Ensure finite
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
