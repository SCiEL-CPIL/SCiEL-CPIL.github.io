import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero or invalid z
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Probability of Improvement (PI) for minimization
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This is more robust than EI in short-budget scenarios and encourages
    # broader exploration of the Rosenbrock valley before exploiting.
    
    # Compute z-score for PI
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement
    pi = norm.cdf(z)
    
    # Cost-aware scaling: reduce the score for high-cost regions
    # Use a mild cost penalty to allow exploration without severe penalty
    cost_penalty = 1.0 / (1.0 + 0.5 * (cost - 1.0))
    
    # Combine PI with cost-aware scaling
    scores = pi * cost_penalty
    
    # Handle any potential non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
