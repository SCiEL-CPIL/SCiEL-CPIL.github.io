"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

# Cache for standard normal CDF and PDF to avoid redundant calculation if possible, 
# though for 2000 points, vectorized scipy is fast enough.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) 
    with a multiplicative exponential cost gate.
    
    Args:
        mu: Array of GP means (shape 2000,)
        sigma: Array of GP standard deviations (shape 2000,)
        f_best: Current best observed function value (scalar)
        cost: Array of evaluation costs (shape 2000,)
        
    Returns:
        Array of acquisition scores (shape 2000,), higher is better.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Note: We are minimizing, so improvement is f_best - mu.
    
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle potential numerical issues where EI might be negative due to floating point errors
    ei = np.maximum(ei, 0.0)
    
    # Cost-awareness:
    # The cost is 1 + mean(x_scaled). The minimum possible cost is 1.0.
    # We apply a multiplicative exponential gate: exp(-beta * (cost - cost_min))
    # This heavily penalizes expensive points.
    # We need to choose beta. If beta is too high, we might ignore good points that are slightly expensive.
    # If beta is too low, it behaves like linear penalty.
    # Let's estimate a reasonable cost range. Costs are in [1, 2].
    # A beta around 5-10 creates a sharp drop-off.
    # Let's use a dynamic beta or a fixed one. Fixed is safer for stability.
    # Let's try beta = 5.0.
    # cost_min is theoretically 1.0.
    
    cost_min = 1.0
    # To make the gate relative to the current population's costs, we could use the median cost,
    # but using the theoretical minimum ensures we always prefer cheaper points if EI is similar.
    # However, if all points are expensive, we still need to pick one.
    # The exponential decay ensures that if EI is high enough, it can overcome the cost penalty.
    
    beta = 5.0
    cost_penalty = np.exp(-beta * (cost - cost_min))
    
    # The score is EI * cost_penalty
    # This is equivalent to EI / exp(beta * (cost - cost_min))
    
    scores = ei * cost_penalty
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
