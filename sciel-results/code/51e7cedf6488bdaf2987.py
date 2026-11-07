import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Calculate Lower Confidence Bound (LCB) score for minimization
    # Score = - (mu - beta*sigma) = -mu + beta*sigma
    beta = 3.0
    lcb_score = -mu + beta * sigma
    
    # Normalize scores to [0, 1]
    eps = 1e-9
    
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + eps
    ei_norm = (ei - ei_min) / ei_range
    
    lcb_min = np.min(lcb_score)
    lcb_max = np.max(lcb_score)
    lcb_range = lcb_max - lcb_min + eps
    lcb_norm = (lcb_score - lcb_min) / lcb_range
    
    # Adaptive Weighting
    # Use the mean sigma to determine the balance between exploration (LCB) and exploitation (EI)
    # Early on, sigma is high -> favor LCB
    # Late on, sigma is low -> favor EI
    
    mean_sigma = np.mean(sigma)
    max_sigma = np.max(sigma)
    
    # Ratio of current uncertainty to maximum uncertainty
    # This will be close to 1 early on, and drop as we fill in the function
    if max_sigma > 1e-9:
        uncertainty_ratio = mean_sigma / max_sigma
    else:
        uncertainty_ratio = 0.0
        
    # Map uncertainty_ratio to a weight for LCB
    # We want w_lcb to be high when uncertainty_ratio is high.
    # w_lcb = 0.5 + 0.5 * uncertainty_ratio
    # This gives w_lcb in [0.5, 1.0].
    # w_ei = 1.0 - w_lcb -> [0.0, 0.5].
    
    w_lcb = 0.5 + 0.5 * uncertainty_ratio
    w_ei = 1.0 - w_lcb
    
    scores = w_lcb * lcb_norm + w_ei * ei_norm
    
    return scores
