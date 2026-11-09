"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lagrangian relaxation of the cost constraint.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero and negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Dynamic Lagrange multiplier lambda
    # We want to balance EI against cost. 
    # A common heuristic is to set lambda proportional to the current EI scale relative to cost.
    # Alternatively, we can tune lambda to target a specific average cost.
    # Here, we use a simple adaptive strategy: 
    # lambda = mean(EI) / mean(cost) * alpha, where alpha is a tuning parameter.
    # This ensures that the penalty term is of similar magnitude to the reward term.
    
    # Calculate a baseline lambda
    mean_ei = np.mean(ei)
    mean_cost = np.mean(cost)
    
    if mean_cost > 0:
        lambda_val = (mean_ei / mean_cost) * 1.0
    else:
        lambda_val = 0.0
    
    # Ensure lambda is non-negative
    lambda_val = max(lambda_val, 0.0)
    
    # Compute Lagrangian score: EI - lambda * cost
    # This decouples the exploration from the explicit division, preserving posterior geometry
    scores = ei - lambda_val * cost
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
