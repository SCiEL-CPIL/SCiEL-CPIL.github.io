import numpy as np
import time
from scipy.stats import norm

_start_time = None
_call_idx = 0

def acquisition(mu, sigma, f_best):
    global _start_time, _call_idx
    if _start_time is None:
        _start_time = time.time()
    
    _call_idx += 1
    
    # Deadline guard
    if time.time() - _start_time > 325:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.where(np.isfinite(ei), ei, 0.0)
        return ei

    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Probability of Improvement
    poi = norm.cdf(z)
    poi = np.where(np.isfinite(poi), poi, 0.0)
    
    # Normalized PI to [0,1]
    poi_min = np.min(poi)
    poi_max = np.max(poi)
    poi_range = poi_max - poi_min
    if poi_range > 1e-12:
        poi_norm = (poi - poi_min) / poi_range
    else:
        poi_norm = np.ones_like(poi)
    
    # Normalized EI to [0,1]
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.ones_like(ei)
    
    # Weighted combination: early calls favor PI (exploration), later calls favor EI (exploitation)
    # The weight on EI increases with call count
    if _call_idx <= 5:
        w_ei = 0.3
    elif _call_idx <= 10:
        w_ei = 0.5
    elif _call_idx <= 15:
        w_ei = 0.7
    else:
        w_ei = 0.9
    
    score = w_ei * ei_norm + (1 - w_ei) * poi_norm
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
