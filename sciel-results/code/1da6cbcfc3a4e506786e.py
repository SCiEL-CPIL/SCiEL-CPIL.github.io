import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a combination of Expected Improvement
    and Lower Confidence Bound, weighted by cost.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP std dev predictions  
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
    - scores: array of shape (2000,) - higher means evaluate next
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid division issues
    sigma = np.clip(sigma, 1e-8, None)
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # Lower Confidence Bound (LCB) for minimization
    # Use a moderate exploration parameter
    xi = 0.01
    lcb = mu - xi * sigma
    
    # Combine EI and LCB: prefer points with high EI but also low predicted value
    # Normalize both to [0, 1] range for fair combination
    ei_min, ei_max = np.min(ei), np.max(ei)
    lcb_min, lcb_max = np.min(lcb), np.max(lcb)
    
    # Handle edge cases where all values are the same
    ei_range = ei_max - ei_min
    lcb_range = lcb_max - lcb_min
    
    if ei_range < 1e-10:
        ei_norm = np.ones_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
    
    if lcb_range < 1e-10:
        lcb_norm = np.ones_like(lcb)
    else:
        lcb_norm = (lcb - lcb_min) / lcb_range
    
    # LCB lower is better, so invert for scoring
    lcb_score = 1 - lcb_norm
    
    # Combine: 70% EI, 30% LCB
    combined = 0.7 * ei_norm + 0.3 * lcb_score
    
    # Cost-aware: divide by cost
    scores = combined / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
