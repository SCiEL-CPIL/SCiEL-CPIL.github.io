import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using cost-adjusted Lower Confidence Bound (LCB).
    
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
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is strictly positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost-adjusted LCB:
    # We want to minimize mu - beta * sigma / cost
    # So we maximize: -mu + beta * sigma / cost
    # 
    # The exploration term (uncertainty bonus) is scaled by 1/cost,
    # which penalizes high-cost regions more heavily in the uncertainty bonus.
    # The mean term remains invariant.
    #
    # Choose beta to balance exploration and exploitation.
    # A typical value for beta is around 2-5 for LCB.
    beta = 3.0
    
    # Calculate the cost-adjusted LCB score
    # score = -mu + beta * sigma / cost
    score = -mu + beta * sigma_safe / cost_safe
    
    # Ensure output is finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
