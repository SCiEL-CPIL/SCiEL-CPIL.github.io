import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    with a Gaussian cost penalty centered at the minimum possible cost.
    
    Parameters:
    mu: (2000,) array of GP means
    sigma: (2000,) array of GP standard deviations
    f_best: float, the lowest observed value
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    (2000,) array of acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Z-scores for Expected Improvement (Minimization)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware penalty
    # Cost ranges from 1 to 2.
    # We apply a Gaussian penalty that is 1.0 at cost=1 and decays as cost increases.
    # This allows high-EI points in slightly more expensive regions to still be considered
    # if their improvement is sufficiently large, unlike hard cutoffs or aggressive power laws.
    
    # Center the penalty at the minimum cost (1.0)
    cost_center = 1.0
    # Width of the Gaussian penalty. 
    # A larger sigma_cost means a softer penalty (more exploration in expensive regions).
    # A smaller sigma_cost means a sharper penalty (strong preference for cheap points).
    # Given the cost range [1, 2], a sigma_cost of 0.8 provides a moderate penalty 
    # (factor ~0.28 at max cost) which balances cost and improvement well.
    sigma_cost = 0.8
    
    cost_excess = cost - cost_center
    # Gaussian decay: exp(-0.5 * (x/sigma)^2)
    cost_penalty = np.exp(-0.5 * (cost_excess / sigma_cost) ** 2)
    
    # Combine EI and cost penalty
    score = ei * cost_penalty
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
