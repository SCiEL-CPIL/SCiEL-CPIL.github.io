import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Hybrid EI/UCB acquisition function.
    
    Combines Expected Improvement (exploitation) with a UCB-style 
    exploration term (sigma), weighted adaptively by the current best 
    value, and penalized by the square root of the evaluation cost 
    to mitigate bias towards cheap, low-quality regions.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviations.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Evaluation costs for each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Safety clamps
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI can be numerically negative or very small; clamp to 0
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration weight
    # Higher f_best (early in optimization) -> more exploration
    # Lower f_best (late in optimization) -> less exploration
    # Clip to prevent extreme values if f_best is huge or negative
    lambda_explore = np.clip(f_best, 0.1, 10.0)
    
    # UCB-like exploration term
    exploration = lambda_explore * sigma_safe
    
    # Combine Exploitation and Exploration
    combined_score = ei + exploration
    
    # Apply softened cost penalty
    # cost is in [1, 2]. sqrt(cost) is in [1, 1.414].
    # This reduces the penalty for higher cost (central) points compared to linear cost.
    penalty = np.sqrt(cost_safe)
    
    scores = combined_score / penalty
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
