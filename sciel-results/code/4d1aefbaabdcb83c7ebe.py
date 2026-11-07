"""Rewritten cost-aware acquisition for levy 2d BO using Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Thompson Sampling.
    
    mu: (2000,) predicted mean in objective units
    sigma: (2000,) predicted std in objective units
    f_best: lowest observed value
    cost: (2000,) cost of evaluating each candidate (1 + mean(x_scaled))
    
    Returns: (2000,) scores, higher = evaluate next
    """
    # Draw a sample from the posterior N(mu, sigma^2)
    # Use standard normal random numbers
    eps = np.random.randn(mu.shape[0])
    sampled_values = mu + sigma * eps
    
    # We want to minimize the sampled value (lower is better)
    # We also want to minimize cost
    # Score: -sampled_value / cost
    # Higher score is better.
    
    cost_safe = np.maximum(cost, 1e-9)
    score = -sampled_values / cost_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Replace -inf with a very low number to avoid issues if argmax is used, 
    # though the caller likely just takes argmax. -inf is fine for comparison.
    # However, the prompt says "Return 2000 finite scores". 
    # So we must replace -inf with a finite number.
    # A very low number like -1e10.
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
