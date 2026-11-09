import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Thompson Sampling.
    
    Args:
        mu: Mean of the GP prediction, shape (N,)
        sigma: Standard deviation of the GP prediction, shape (N,)
        f_best: Best observed value so far.
        cost: Cost of evaluating each point, shape (N,)
        
    Returns:
        scores: Acquisition scores, shape (N,), higher = evaluate next
    """
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin before 330s kill
    
    # Input validation and safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure shapes are correct
    n = mu.shape[0]
    
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Check for non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1.0)
    
    # Thompson Sampling for minimization
    # Sample from the posterior distribution N(mu, sigma^2)
    # We want to find points with low function values.
    # A sample 's' is a potential function value.
    # We want to maximize the "desirability", which is inversely related to the function value.
    # So we use -s.
    # We also divide by cost to favor cheaper points.
    
    # Generate random samples
    # Using np.random.normal is efficient for arrays
    samples = np.random.normal(mu, sigma_safe)
    
    # Calculate scores
    # Higher score = better
    # Low function value (low sample) -> high score
    # Low cost -> high score
    scores = -samples / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return a safe fallback: just use -mu/cost (UCB with beta=0)
        fallback_scores = -mu / cost_safe
        fallback_scores = np.where(np.isfinite(fallback_scores), fallback_scores, -1e10)
        return fallback_scores
    
    return scores
