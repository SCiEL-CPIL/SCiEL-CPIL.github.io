"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
# Safety margin: 10 seconds before the 330s limit
_DEADLINE = _START_TIME + 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    Args:
        mu: Array of mean predictions (shape 2000,).
        sigma: Array of standard deviation predictions (shape 2000,).
        f_best: Scalar, best observed value.
        cost: Array of costs for evaluating each candidate (shape 2000,).
        
    Returns:
        Array of scores (shape 2000,). Higher is better.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return random scores to break ties or just zeros
        # Using zeros might cause the evaluator to pick the first one, which is arbitrary.
        # Random is safer to avoid bias if the loop continues briefly.
        return np.random.rand(len(mu))

    # UCB for minimization:
    # We want to maximize the lower confidence bound of improvement.
    # Improvement = f_best - f(x)
    # Lower bound of f(x) is mu(x) - beta * sigma(x)
    # So we maximize: f_best - (mu(x) - beta * sigma(x)) = f_best - mu(x) + beta * sigma(x)
    
    # Exploration parameter
    beta = 2.5
    
    # Ensure sigma is not zero to avoid numerical issues, though UCB is linear
    # so it's less sensitive than EI, but good practice.
    # Actually, if sigma is 0, the term is 0, which is fine.
    
    # Calculate scores
    # Vectorized operation
    scores = (f_best - mu + beta * sigma) / cost
    
    # Handle potential infinities or NaNs
    # If cost is 0 (shouldn't happen, min cost is 1), division by zero.
    # If mu or sigma are inf/nan.
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
