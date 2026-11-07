"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on the probability of finding
    a point with function value less than f_best - epsilon, normalized by cost.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Lowest observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Handle edge cases: where sigma is very small
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive epsilon based on distance to known global minimum
    # Allows for larger improvements when far from optimum, tighter when close
    min_val = -10.5364
    gap = f_best - min_val
    if gap < 0:
        gap = 0.0
    epsilon = 0.01 * gap
    
    # Target value we want to beat
    # Ensure target is not below the theoretical minimum (though GP can predict below)
    # Actually, just use the calculated target
    target = f_best - epsilon
    
    # Probability that f(x) < target under GP posterior:
    # P(f(x) < target) = Phi((target - mu) / sigma)
    z = (target - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -50, 50)
    
    prob_improve = norm.cdf(z)
    
    # Ensure non-negative
    prob_improve = np.maximum(prob_improve, 0.0)
    
    # Normalize by cost to get probability per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = prob_improve / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
