# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Adaptive UCB per unit cost.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviations in objective units.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Evaluation costs for each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost is guaranteed positive (1 + mean(x scaled to [0,1]))
    cost_safe = np.maximum(cost, 1e-9)
    
    # Adaptive kappa: reduces exploration as f_best approaches the minimum (0)
    # f_best starts high and decreases.
    # kappa = 1.0 + 1.0 * f_best
    # If f_best ~ 10, kappa ~ 11 (explore)
    # If f_best ~ 0, kappa ~ 1 (exploit)
    kappa = 1.0 + 1.0 * f_best
    
    # UCB for minimization: maximize -mu + kappa * sigma
    ucb = -mu + kappa * sigma_safe
    
    # Cost-aware: divide by cost
    scores = ucb / cost_safe
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
