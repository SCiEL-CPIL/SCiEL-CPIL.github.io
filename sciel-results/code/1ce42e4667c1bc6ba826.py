"""Cost-aware acquisition for Styblinski-Tang 2D BO."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Pre-compute some constants for speed
# We use a hybrid approach:
# 1. Cost-aware Expected Improvement (EI)
# 2. Cost-aware Upper Confidence Bound (UCB)
# 3. A small amount of random exploration to avoid local optima

def acquisition(mu, sigma, f_best, cost):
    """
    Compute acquisition scores for 2000 candidate points.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Start with a deadline guard
    t_start = time.time()
    deadline = 280.0  # 300s limit with safety margin
    
    n = len(mu)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Upper Confidence Bound
    # Use a beta parameter that balances exploration/exploitation
    beta = 2.0
    ucb = mu + beta * sigma
    
    # For minimization, we want to minimize, so UCB should be inverted
    # Actually, for minimization with GP, we want to find low values
    # The acquisition should favor low mu and high sigma
    # EI already does this. Let's use a cost-aware version.
    
    # Cost normalization: cost is in [1, 2] range based on the problem
    # cost = 1 + mean(x scaled to [0,1])
    # So cost is in [1, 2]
    
    # Cost-aware EI: divide by cost
    cost_ei = ei / cost
    
    # Also consider a cost-aware UCB variant
    # For minimization, we want low predicted values
    # A good cost-aware metric: (f_best - mu) / cost + beta * sigma / cost
    # This rewards points that are predicted to be much better than f_best,
    # normalized by cost, plus an exploration term also normalized by cost
    
    # Let's use a combination:
    # score = alpha * cost_ei + (1 - alpha) * cost_ucb
    # where cost_ucb = (f_best - mu + beta * sigma) / cost
    
    # Actually, let's think more carefully.
    # EI is a good measure of expected improvement.
    # Dividing by cost gives expected improvement per unit cost.
    # This is a reasonable acquisition function.
    
    # But EI can be zero when mu > f_best (i.e., when the point is predicted
    # to be worse than the best). In that case, we might still want to explore
    # if sigma is high.
    
    # Let's use a hybrid:
    # Primary: cost-aware EI
    # Secondary: if EI is very small, add a small exploration bonus based on sigma/cost
    
    # Exploration bonus: sigma / cost, scaled down
    exploration_bonus = 0.1 * sigma / cost
    
    # Combine
    score = cost_ei + exploration_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() - t_start > deadline:
        return score
    
    # Return the scores
    return score
# EVOLVE-BLOCK-END
