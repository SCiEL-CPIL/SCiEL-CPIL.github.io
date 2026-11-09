import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Quantile-Based Cost-Discounted Lower Confidence Bound acquisition.
    
    Uses a dynamic quantile (5th-20th percentile) of the predictive distribution
    instead of linear sigma, combined with a sigmoidal cost penalty.
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    
    # Dynamic quantile selection based on exploration-exploitation balance
    # Use the 10th percentile of the predictive distribution (aggressive exploitation)
    # but adapt based on how uncertain the model is
    mu_std = np.std(mu)
    sigma_mean = np.mean(sigma)
    
    # If uncertainty is high relative to mean spread, explore more
    if mu_std < 1e-8:
        # Very flat prediction, explore more
        quantile_level = 0.02
    else:
        uncertainty_ratio = sigma_mean / (mu_std + 1e-9)
        # Map uncertainty ratio to quantile level:
        # High uncertainty -> lower quantile (more exploration)
        # Low uncertainty -> higher quantile (more exploitation)
        quantile_level = 0.20 - 0.18 * np.tanh(uncertainty_ratio / 2.0)
        quantile_level = np.clip(quantile_level, 0.02, 0.15)
    
    # Compute z-value for the chosen quantile (lower tail)
    # For a normal distribution, the q-th quantile is at z = norm.ppf(q)
    z_quantile = norm.ppf(quantile_level)
    
    # Quantile-based lower bound: mu + z_quantile * sigma
    # Since z_quantile is negative, this gives a lower estimate
    q_lb = mu + z_quantile * sigma
    
    # Expected improvement relative to f_best
    improvement = f_best - q_lb
    improvement = np.maximum(improvement, 0.0)
    
    # Sigmoidal cost discount: penalizes high-cost candidates smoothly
    # cost ranges from 1 (min) to 2 (max) since cost = 1 + mean(x in [0,1])
    # Use a sigmoid that discounts scores for expensive candidates
    # Center the sigmoid around cost=1.5 with steep enough slope
    cost_center = 1.3
    cost_scale = 0.15
    cost_discount = 1.0 / (1.0 + np.exp((cost - cost_center) / cost_scale))
    
    # Combine: quantile-based improvement * cost discount
    scores = improvement * cost_discount
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
