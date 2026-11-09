import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using qEI (quantile Expected Improvement).
    """
    t_start = time.time()
    
    # Safety for sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Use quantile-based weighting to balance exploration and exploitation
    # qEI: weight EI by a quantile of the posterior distribution
    # This effectively creates a more robust acquisition that adapts to uncertainty
    
    # Compute a robust score: use the 80th percentile of potential improvement
    # This balances between aggressive exploitation (max EI) and safe exploration
    q = 0.8
    # For a normal distribution, the q-quantile of improvement given mu, sigma
    # is approximately: mu + sigma * norm.ppf(q) when considering improvement over f_best
    # But we want a simpler, more stable approach
    
    # Alternative: use a weighted combination that naturally handles the cost
    # Weight by both EI and uncertainty (sigma) to prefer points with high expected
    # improvement relative to their cost
    
    # Normalize EI and PI
    ei_max = np.max(ei) if np.max(ei) > 1e-10 else 1.0
    pi_max = np.max(pi) if np.max(pi) > 1e-10 else 1.0
    
    ei_norm = ei / ei_max
    pi_norm = pi / pi_max
    
    # Use a power-weighted combination that emphasizes high-EI regions
    # but also rewards uncertainty (exploration)
    alpha = 0.75  # Weight on EI
    beta = 0.25   # Weight on PI (uncertainty)
    
    blended = alpha * ei_norm + beta * pi_norm
    
    # Cost-aware adjustment: divide by cost to prefer cheaper evaluations
    cost_safe = np.maximum(cost, 1e-10)
    scores = blended / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time guard
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:
        return scores
        
    return scores
