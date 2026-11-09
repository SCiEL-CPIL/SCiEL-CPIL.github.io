import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware hybrid acquisition function combining Expected Improvement (EI)
    and an exploration term based on standard deviation (UCB-like), normalized by cost.
    
    Parameters:
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviation predictions
    f_best: float - best observed function value
    cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    # Ensure cost is positive to avoid division issues
    cost_safe = np.maximum(cost, 1e-9)
    
    # Ensure sigma is positive for z calculation
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Exploration term
    # We use a fixed beta for simplicity and stability, as adaptive beta
    # without iteration count can be tricky. A value of 1.0 balances
    # exploitation and exploration well for this problem size.
    beta = 1.0
    
    # Hybrid Score: (EI + Exploration) / Cost
    score = (ei + beta * sigma) / cost_safe
    
    # Replace any non-finite values (NaN, Inf) with 0.0 to ensure validity
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
