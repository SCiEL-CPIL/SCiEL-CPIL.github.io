"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI)
    # f_best is the current best (lowest) value
    # We want to minimize, so improvement is f_best - mu
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Progress-adaptive lambda
    # As f_best decreases (improves), lambda increases, penalizing cost more heavily
    # f_best starts potentially high (e.g., 10+) and goes to 0
    # If f_best is large, lambda is small (explore cheaply but not too aggressively)
    # If f_best is small, lambda is large (exploit, save cost)
    # Using a smooth mapping
    lambda_val = 2.0 / (1.0 + np.clip(f_best, 0.0, 10.0))
    
    # Cost penalty
    # cost is in [1, 2]. We shift by 1 so min cost has penalty 0.
    # score = EI * exp(-lambda * (cost - 1))
    # This favors low cost, with strength controlled by lambda
    
    # To avoid underflow/overflow issues, we can just compute the ratio or product
    # Since EI can be very small, and exp can be small, we return the product.
    # However, if EI is 0, score is 0.
    
    cost_shifted = cost - 1.0
    penalty = np.exp(-lambda_val * cost_shifted)
    
    scores = ei * penalty
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
