import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Cost-Adjusted Upper Confidence Bound (UCB) for minimization.
    
    Parameters:
    mu: array of shape (2000,) - GP mean predictions in objective units
    sigma: array of shape (2000,) - GP standard deviations in objective units
    f_best: float - lowest observed value so far
    cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma and cost are positive to avoid division by zero or negative variances
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # UCB parameter for exploration
    # Since we don't have the number of iterations, we use a fixed value.
    # 2.0 is a reasonable starting point for balancing exploitation and exploration.
    beta = 2.0
    
    # For minimization, we want low mu. 
    # The term -mu rewards low predictions.
    # The term beta * sigma rewards high uncertainty.
    # We maximize this quantity, divided by cost.
    scores = (-mu + beta * sigma) / cost
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
