"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Args:
        mu: Mean of the GP posterior (shape: N,)
        sigma: Standard deviation of the GP posterior (shape: N,)
        f_best: Best observed function value (scalar)
        cost: Cost of evaluating each candidate (shape: N,)
        
    Returns:
        scores: Acquisition scores (shape: N,), higher is better.
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Ensure cost is positive to avoid division by zero
    cost_safe = np.maximum(cost, 1e-8)
    
    # Draw samples from the posterior distribution
    # This introduces stochasticity that helps escape local optima in the acquisition landscape
    noise = np.random.normal(0, 1, size=mu.shape)
    sampled_values = mu + sigma * noise
    
    # Calculate the potential improvement over the current best
    improvements = f_best - sampled_values
    
    # Score is improvement per unit cost
    scores = improvements / cost_safe
    
    # Replace any non-finite values with 0 to ensure validity
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
