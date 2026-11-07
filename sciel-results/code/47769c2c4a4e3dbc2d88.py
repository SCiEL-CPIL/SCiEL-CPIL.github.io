"""New program: Cost-Aware Upper Confidence Bound (CA-UCB).
This approach uses the Upper Confidence Bound (optimistic bound) for minimization.
It balances exploitation (low mean) and exploration (high variance) using a beta parameter.
The score is the optimistic bound divided by the evaluation cost, favoring
high-potential, low-cost points. This is distinct from Expected Improvement
approaches as it does not integrate the posterior tail but uses a heuristic bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Aware Upper Confidence Bound (CA-UCB) acquisition function.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Beta parameter for exploration-exploitation trade-off
    # A value of 2.0 is a common heuristic for 2D problems with Matern-5/2
    beta = 2.0
    
    # For minimization, we want to find low values.
    # The "optimistic" value (lower confidence bound) is mu - beta * sigma.
    # We want to MAXIMIZE the acquisition score, so we negate the LCB:
    # Score_base = -(mu - beta * sigma) = -mu + beta * sigma
    scores_base = -mu + beta * sigma
    
    # Cost-aware adjustment: Divide by cost
    # This favors points with high optimistic improvement potential per unit cost
    scores = scores_base / (cost + 1e-8)
    
    # Handle any numerical issues
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Shift to ensure non-negative scores for stability (optional but safe)
    # Note: UCB scores can be negative if mu is very high. Shifting preserves relative order.
    scores = scores - np.min(scores)
    
    return scores
# EVOLVE-BLOCK-END
