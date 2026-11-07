"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a cost-awareness penalty.
    
    The acquisition score is:
        EI / cost^alpha
    
    where EI is the standard Expected Improvement and alpha is a cost
    sensitivity parameter. We also add a small exploration bonus based on
    uncertainty to encourage sampling in uncertain regions, weighted by
    the cost efficiency.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions at candidate points (in objective units)
    sigma : array of shape (2000,)
        GP standard deviation predictions at candidate points (in objective units)
    f_best : float
        Best (lowest) observed objective value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-awareness: divide by cost raised to a power
    # alpha = 0.5 balances exploitation and cost efficiency
    alpha = 0.5
    cost_safe = np.maximum(cost, 1e-9)
    cost_factor = cost_safe ** alpha
    
    # Base acquisition: EI per unit cost
    acq_base = ei / cost_factor
    
    # Add a small exploration bonus weighted by uncertainty and cost efficiency
    # This encourages exploring high-uncertainty regions when they are cheap
    # The bonus is proportional to sigma / cost, but scaled down to not dominate
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * sigma_safe / cost_safe
    
    # Combine
    scores = acq_base + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
