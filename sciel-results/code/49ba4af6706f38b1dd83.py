"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition function.
    
    Args:
        mu: Array of GP means (shape 2000,).
        sigma: Array of GP standard deviations (shape 2000,).
        f_best: Scalar, lowest observed value.
        cost: Array of evaluation costs (shape 2000,).
        
    Returns:
        Array of scores (shape 2000,). Higher scores are preferred.
    """
    # Sample a single standard normal variable
    # Using a fixed seed is not possible here as the function must be stateless or random per call?
    # The prompt doesn't specify if acquisition should be deterministic. 
    # Standard BO acquisition functions are deterministic given the inputs.
    # However, Thompson Sampling is inherently stochastic.
    # If the evaluator calls acquisition multiple times in a loop, it should get different samples.
    # If the evaluator expects a deterministic function for reproducibility of the "best candidate" selection,
    # pure TS might be tricky if the "best" changes every time the function is called.
    # But the task says: "the evaluator scores 2000 new uniform candidates with acquisition and evaluates the best one."
    # This implies a single call to acquisition per iteration.
    # So we just need to return a vector of scores.
    
    # To ensure we don't hit zero division or issues, we assume cost > 0.
    # cost is 1 + mean(x_scaled), so min cost is 1.
    
    # Draw a random sample from N(0, 1)
    # We use numpy's default RNG which is seeded by the system.
    r = np.random.normal(0, 1)
    
    # Sample theta
    theta = mu + sigma * r
    
    # Score: higher is better. We want to minimize f, so we want low theta.
    # We divide by cost to account for expense.
    # score = -theta / cost
    
    # Ensure finite scores
    scores = -theta / cost
    
    # Handle any potential NaNs or infs (though sigma and mu should be finite from GP)
    # If sigma is 0, theta = mu.
    # If cost is 0 (unlikely), we'd have issues. Cost >= 1.
    
    return scores
# EVOLVE-BLOCK-END
