import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using adaptive Probability of Improvement (PI).
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores (higher is better).
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    # Using standard normal CDF
    pi = norm.cdf(z)
    
    # Adaptive exploration-exploitation tradeoff based on f_best
    # When f_best is high (early), explore more; when low (late), exploit more
    # Rastrigin minimum is 0, so f_best approaching 0 means we're close
    # Scale beta from ~0.5 (exploratory) to ~2.0 (exploitative)
    # Use a smooth transition based on how close f_best is to 0
    # f_best ranges roughly from ~10+ (early) to ~0.1 (late)
    # Map this to beta in [0.5, 2.0]
    f_best_clipped = np.clip(f_best, 0.0, 10.0)
    # When f_best is 10, beta=0.5; when f_best is 0, beta=2.0
    # Linear interpolation: beta = 0.5 + 1.5 * (1 - f_best/10)
    beta = 0.5 + 1.5 * (1.0 - f_best_clipped / 10.0)
    # Clamp beta to reasonable range
    beta = np.clip(beta, 0.3, 3.0)
    
    # Add a small exploration bonus for high-uncertainty regions
    # This helps escape local optima in multimodal landscapes
    # Use a normalized uncertainty term
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 1e-9 else 1.0
    uncertainty_bonus = 0.05 * (sigma_safe / sigma_max)
    
    # Cost-aware score: Maximize (PI^beta + uncertainty_bonus) per unit cost
    # Raising PI to beta makes it more selective when beta > 1
    pi_powered = pi ** beta
    
    # Combine improvement probability with exploration bonus
    combined = pi_powered + uncertainty_bonus
    
    # Divide by cost for cost-awareness
    score = combined / cost
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
