"""Cost-aware acquisition using Expected Improvement (EI) with cost normalization."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on Cost-Adjusted Lower Confidence Bound (LCB).
    
    This approach uses a linear combination of the mean and standard deviation to
    estimate the potential improvement, which encourages exploration in regions
    with high uncertainty. The score is normalized by the evaluation cost.
    
    Args:
        mu: Array of GP means, shape (2000,)
        sigma: Array of GP standard deviations, shape (2000,)
        f_best: Scalar, lowest observed value.
        cost: Array of costs for each candidate, shape (2000,)
        
    Returns:
        Array of scores, shape (2000,), higher is better.
    """
    epsilon = 1e-9
    
    # Handle zero or near-zero sigma
    sigma_safe = np.maximum(sigma, epsilon)
    
    # Exploration parameter beta
    # A higher beta encourages more exploration.
    # For Ackley 2D with multiple local minima, beta=2.0 is a reasonable start.
    beta = 2.0
    
    # Lower Confidence Bound for minimization:
    # We estimate the best possible value at x as mu - beta * sigma.
    # The potential improvement over f_best is f_best - (mu - beta * sigma).
    lcb = mu - beta * sigma_safe
    potential_improvement = f_best - lcb
    
    # We only care about points that are likely to improve upon f_best.
    # If potential_improvement <= 0, the score is 0.
    potential_improvement = np.maximum(potential_improvement, 0.0)
    
    # Cost-aware score: Potential improvement divided by cost
    scores = potential_improvement / (cost + epsilon)
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
