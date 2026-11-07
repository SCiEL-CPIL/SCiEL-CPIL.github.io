"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization minimization.
    
    Uses a pure UCB strategy: score = (-mu + kappa * sigma) / cost^gamma
    with high gamma to aggressively favor low-cost, high-uncertainty regions.
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Parameters tuned for cost-aware exploration
    gamma = 3.0
    kappa = 2.5
    
    # UCB score for minimization: (-mu + kappa * sigma)
    # Higher score means lower predicted value and/or higher uncertainty
    ucb_score = (-mu + kappa * sigma_safe)
    
    # Normalize by cost^gamma to strongly favor low-cost evaluations
    final_score = ucb_score / (cost_safe ** gamma)
    
    # Ensure all values are finite
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
