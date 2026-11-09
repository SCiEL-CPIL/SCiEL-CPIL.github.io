import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    mu : np.ndarray, shape (2000,)
        Mean of the GP at the candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP at the candidate points.
    f_best : float
        Best observed value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate point.
        
    Returns:
    np.ndarray, shape (2000,)
        Acquisition scores for each candidate point.
    """
    # Avoid division by zero in z-score
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Normalization to [0, 1] to prevent outliers from dominating
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei - ei_min) / ei_range
    
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    sigma_range = sigma_max - sigma_min + 1e-9
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Adaptive weights based on average uncertainty
    # If mean sigma is high, we explore more (higher weight on sigma).
    # If mean sigma is low, we exploit more (higher weight on EI).
    mean_sigma = np.mean(sigma)
    
    # Scale mean_sigma to a weight between 0.1 and 0.5
    # This ensures some exploration always, but allows exploitation to dominate later.
    w_sigma = 0.1 + 0.4 * np.clip(mean_sigma * 2.0, 0.0, 1.0)
    w_ei = 1.0 - w_sigma
    
    blended_score = w_ei * ei_norm + w_sigma * sigma_norm
    
    # Cost adjustment
    # We divide by cost to favor cheaper points.
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = blended_score / cost_safe
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
