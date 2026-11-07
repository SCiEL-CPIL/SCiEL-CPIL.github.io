import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB).
    
    Parameters:
    mu: array-like, shape (2000,), mean of GP
    sigma: array-like, shape (2000,), std of GP
    f_best: float, best observed value
    cost: array-like, shape (2000,), cost of evaluating each point
    
    Returns:
    scores: array, shape (2000,), acquisition scores (higher is better)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure inputs are 1-D and handle non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 1e-12), cost, 1.0)
    
    # Clip sigma to prevent numerical issues
    sigma = np.clip(sigma, 1e-8, 1e6)
    
    # Lower Confidence Bound (LCB) for minimization
    # We want to find points with low predicted values and high uncertainty.
    # LCB = mu - kappa * sigma
    # The "score" is the potential improvement over f_best, adjusted for cost.
    # Potential Improvement = f_best - LCB = f_best - (mu - kappa * sigma) = f_best - mu + kappa * sigma
    
    # Hyperparameter for exploration
    kappa = 1.5
    
    lcb = mu - kappa * sigma
    potential_improvement = f_best - lcb
    
    # Cost-aware scaling
    # Higher cost reduces the score
    scores = potential_improvement / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
