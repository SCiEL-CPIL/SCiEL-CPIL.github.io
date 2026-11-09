import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (POI)
    with dynamic uncertainty-normalized exploration.
    
    Parameters:
    - mu: array of shape (2000,) with GP mean predictions
    - sigma: array of shape (2000,) with GP standard deviation predictions
    - f_best: float, the best (lowest) observed value so far
    - cost: array of shape (2000,) with evaluation costs
    
    Returns:
    - array of shape (2000,) with acquisition scores (higher = better)
    """
    # Ensure inputs are arrays and handle numerical stability
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Small epsilon to avoid division by zero
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # 1. Compute Probability of Improvement (POI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Ensure POI is in [0, 1] (should be, but clip for safety against numerical errors)
    poi = np.clip(poi, 0.0, 1.0)
    
    # 2. Dynamic Exploration Factor
    # We want to explore regions with high uncertainty relative to the global uncertainty.
    # This makes the exploration term adaptive: if the GP is very uncertain everywhere,
    # the relative bonus is normalized. If uncertainty is low everywhere, high-uncertainty
    # points stand out more.
    
    # Calculate a reference uncertainty level (e.g., 95th percentile of sigma)
    # This scales the exploration bonus so it's not dominated by absolute sigma values
    # which might change as the GP trains.
    sigma_ref = np.percentile(sigma_safe, 95)
    if sigma_ref < eps:
        sigma_ref = eps
        
    # Relative uncertainty: how uncertain is this point compared to the most uncertain regions?
    rel_uncertainty = sigma_safe / sigma_ref
    
    # Exploration weight
    # We use a coefficient to balance exploitation (POI) vs exploration (uncertainty).
    # Since POI is bounded [0,1] and rel_uncertainty is typically around [0, 1] (with outliers >1),
    # a weight of 0.5 provides a reasonable balance.
    exploration_weight = 0.5
    
    exploration_term = exploration_weight * rel_uncertainty
    
    # 3. Cost-Aware Scoring
    # We want to maximize (Improvement + Exploration) per unit cost.
    # Score = (POI + Exploration) / Cost
    
    raw_score = poi + exploration_term
    score = raw_score / cost_safe
    
    # Handle any remaining non-finite values (e.g., if cost was 0 or negative, though cost_safe handles 0)
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
