import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Parameters:
    mu: array of shape (N,), mean of GP
    sigma: array of shape (N,), std dev of GP
    f_best: float, best observed value so far
    cost: array of shape (N,), cost of evaluating each candidate
    
    Returns:
    scores: array of shape (N,), acquisition scores (higher is better)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero, though cost >= 1
    cost = np.maximum(cost, 1e-6)
    
    # LCB for minimization: mu - beta * sigma
    # We want to maximize the acquisition score, so we negate the LCB value.
    # Score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    
    # Adaptive Beta:
    # Start with a moderate exploration factor.
    # A dynamic beta can be based on the current uncertainty relative to the best value.
    # However, a fixed beta is often robust for small budgets.
    # Let's use beta = 1.0 as a baseline, which is standard for LCB.
    beta = 1.0
    
    # Calculate LCB scores
    lcb_values = mu - beta * sigma
    scores = -lcb_values / cost
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
