import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive and non-zero to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Inverted UCB (for minimization, maximize -mu + beta * sigma)
    # Beta controls exploration. 2.0 is a reasonable starting point.
    beta = 2.0
    ucb_score = -mu + beta * sigma_safe
    
    # 3. Normalize both scores to [0, 1]
    # Add small epsilon to avoid division by zero
    eps = 1e-9
    
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < eps:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
        
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min
    if ucb_range < eps:
        ucb_norm = np.zeros_like(ucb_score)
    else:
        ucb_norm = (ucb_score - ucb_min) / ucb_range
        
    # 4. Combine
    # Weights: 0.6 for EI, 0.4 for UCB
    w_ei = 0.6
    w_ucb = 0.4
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    return scores
