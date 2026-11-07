"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    
    This function implements a cost-aware UCB strategy for Bayesian Optimization.
    UCB balances exploration (uncertainty) and exploitation (expected value)
    and is known to be robust in multimodal landscapes like Styblinski-Tang.
    The score is normalized by the evaluation cost to prioritize cheaper candidates
    that offer high potential improvement.
    """
    # Convert inputs to numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid division issues in downstream logic
    # and to represent a valid standard deviation
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Ensure cost is strictly positive
    cost_safe = np.maximum(cost, 1e-8)
    
    # UCB Parameter (beta)
    # Controls the trade-off between exploitation and exploration.
    # Higher beta encourages exploring regions with high uncertainty.
    # For multimodal functions, a higher beta helps escape local minima.
    beta = 4.0
    
    # Calculate UCB score for minimization
    # We want to minimize f(x).
    # The GP mean mu approximates f(x). Lower mu is better.
    # The GP std sigma approximates uncertainty. Higher sigma is better for exploration.
    # UCB(x) = -mu(x) + beta * sigma(x)
    # Note: We negate mu because we are minimizing.
    ucb_term = -mu + beta * sigma_safe
    
    # Cost-aware scoring
    # We divide by cost to maximize the "value" per unit cost.
    # This encourages the algorithm to select points that are both promising
    # and inexpensive to evaluate.
    score = ucb_term / cost_safe
    
    # Ensure all returned scores are finite
    # Replace any non-finite values (inf, nan) with -inf to deprioritize them
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
