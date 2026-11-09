"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Pure Upper Confidence Bound (UCB) acquisition function for Bayesian optimization.
    
    Strategy:
    - Uses a high, static exploration coefficient (beta = 5.0) to aggressively explore.
    - For minimization, UCB is defined as: score = mu - beta * sigma.
    - High beta prioritizes regions with high uncertainty (high sigma) over regions with low mean (exploitation).
    - This is structurally different from POI, which uses a CDF transform that compresses high sigma values,
      potentially causing the search to get stuck in local optima by not exploring distant minima enough.
    - The goal is to quickly map the entire domain of Branin 2D, which has multiple local minima,
      by forcing the algorithm to visit unexplored regions where sigma is high.
    """
    # Ensure sigma is strictly positive to avoid numerical issues, though it should be positive by definition
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Static exploration coefficient
    # Branin is multimodal; high beta ensures we don't get trapped in the first local minimum found.
    beta = 5.0
    
    # Compute UCB scores for minimization
    # We want to minimize f(x), so we maximize the score.
    # Score = mu - beta * sigma
    ucb_scores = mu - beta * sigma_safe
    
    # Ensure finite scores
    ucb_scores = np.where(np.isfinite(ucb_scores), ucb_scores, -np.inf)
    
    return ucb_scores
# EVOLVE-BLOCK-END
