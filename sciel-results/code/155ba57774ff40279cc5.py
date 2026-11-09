import numpy as np
from scipy.stats import norm

def _min_max_normalize(arr):
    """Normalize array to [0, 1] range."""
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    arr_range = arr_max - arr_min
    if arr_range < 1e-9:
        return np.ones_like(arr)
    return (arr - arr_min) / arr_range

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute EI
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute LCB (to be maximized)
    beta = 2.0
    lcb_max = beta * sigma - mu
    
    # Thompson Sampling proxy: normalized improvement heuristic
    # Encourages sampling points with high potential relative to uncertainty
    ts_score = (f_best - mu) / (sigma + 1e-9)
    
    # Normalize all components to [0, 1]
    ei_norm = _min_max_normalize(ei)
    lcb_norm = _min_max_normalize(lcb_max)
    ts_norm = _min_max_normalize(ts_score)
    
    # Hybrid score: blend EI, LCB, and Thompson Sampling proxy
    # Weight towards TS for better exploration relative to uncertainty
    score = 0.4 * ei_norm + 0.3 * lcb_norm + 0.3 * ts_norm
    
    return score
