import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB).
    The exploration term is scaled by inverse cost, creating a non-uniform
    exploration landscape that prioritizes high-uncertainty, low-cost areas.
    
    Parameters:
    mu: (N,) array of posterior means.
    sigma: (N,) array of posterior standard deviations.
    f_best: float, lowest observed value.
    cost: (N,) array of evaluation costs.
    
    Returns:
    (N,) array of scores (higher is better).
    """
    # Ensure arrays are floats and 1D
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle potential NaNs or Infs in inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e6, neginf=1e-9)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Ensure cost is positive to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # LCB-based score for minimization:
    # We want to minimize, so lower mu is better.
    # score = -mu + beta * sigma / cost
    # The mean term (-mu) is invariant to cost (exploitation).
    # The exploration term (sigma) is scaled by 1/cost, penalizing
    # high-cost regions more heavily in the uncertainty bonus.
    
    # Beta controls exploration intensity. For short budgets, 
    # aggressive exploration is beneficial.
    beta = 2.0
    
    score = -mu + beta * sigma_safe / cost_safe
    
    # Ensure output is finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
