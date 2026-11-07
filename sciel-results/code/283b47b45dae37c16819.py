# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on rosenbrock 2d.
    
    Uses a combination of:
    - Cost-adjusted LCB for exploration (exploit high uncertainty in cheap regions)
    - Expected Improvement for exploitation
    - Adaptive beta based on uncertainty level
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate (1 + mean(x scaled to [0,1]))
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if mu.size == 0:
        return np.array([])
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Clip cost to avoid division by zero
    cost = np.maximum(cost, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-adjusted EI
    ei_cost = ei / cost
    
    # Adaptive beta: increase exploration when uncertainty is high relative to improvement
    # Use a heuristic based on the ratio of sigma to (f_best - mu)
    improvement = np.maximum(f_best - mu, 1e-9)
    uncertainty_ratio = sigma / improvement
    
    # Beta increases with uncertainty but is bounded
    beta = 1.0 + 0.5 * np.clip(uncertainty_ratio, 0, 5)
    
    # Cost-adjusted LCB
    lcb_cost = (-mu + beta * sigma) / cost
    
    # Combine: weight EI more when we have good predictions, LCB more when uncertain
    # Use a simple blending factor based on how close mu is to f_best
    closeness = np.exp(-improvement / (sigma + 1e-9))
    
    # Final score: blend of cost-adjusted EI and LCB
    scores = 0.5 * ei_cost + 0.5 * lcb_cost
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
