"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Parameters:
    mu: array of shape (N,) - GP mean predictions
    sigma: array of shape (N,) - GP standard deviation predictions
    f_best: float - Best observed objective value
    cost: array of shape (N,) - Cost of evaluating each candidate
    
    Returns:
    scores: array of shape (N,) - Acquisition scores (higher is better)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Handle edge cases where sigma is very small or cost is zero
    # Add small epsilon to avoid division by zero or log(0)
    eps = 1e-8
    
    # Sample from the posterior distribution: x ~ N(mu, sigma^2)
    # We use a single sample for each candidate (vectorized)
    # Note: In a true Thompson Sampling loop, we would sample a function.
    # Here, we approximate by sampling the value at each candidate point independently.
    # This is valid because the GP is Gaussian, and the marginal distribution at each point is Gaussian.
    samples = mu + sigma * np.random.randn(mu.shape)
    
    # For minimization, we want to find points with low sampled values.
    # The cost-aware metric is: sampled_value / cost.
    # Lower is better for this metric.
    # To fit the "higher is better" convention of the acquisition function return,
    # we return the negative of this metric.
    
    # Avoid division by zero in cost
    safe_cost = np.maximum(cost, eps)
    
    # Compute the cost-normalized sampled objective
    # We subtract a large constant or just negate to make lower values correspond to higher scores.
    # Actually, simply negating the ratio works:
    # score = - (sampled_value / cost)
    
    scores = - (samples / safe_cost)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
