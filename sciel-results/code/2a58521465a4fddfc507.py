import numpy as np
import time

# Pre-compute lookup table for Weibull shape parameter k from CV
# CV = sqrt(Var) / Mean
# Var = lambda^2 * (Gamma(1+2/k) - Gamma(1+1/k)^2)
# Mean = lambda * Gamma(1+1/k)
# CV = sqrt(Gamma(1+2/k) - Gamma(1+1/k)^2) / Gamma(1+1/k)

from scipy.special import gamma

_k_values = np.arange(0.5, 5.0, 0.1)
_cv_values = np.array([
    np.sqrt(gamma(1 + 2/k) - gamma(1 + 1/k)**2) / gamma(1 + 1/k) 
    for k in _k_values
])

# Globals for state
_items_seen = []
_last_fit_len = 0
_k_est = 1.0  # Default shape
_lambda_est = 10.0 # Default scale

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _items_seen, _last_fit_len, _k_est, _lambda_est
    
    start_time = time.time()
    
    # Add current item to history
    _items_seen.append(item)
    
    # Refit parameters periodically
    if len(_items_seen) > 50 and (len(_items_seen) - _last_fit_len > 20):
        # Check time budget before doing heavy work
        if time.time() - start_time < 10.0:
            try:
                items_arr = np.array(_items_seen)
                mu = np.mean(items_arr)
                sigma = np.std(items_arr)
                
                if mu > 0 and sigma > 0:
                    cv = sigma / mu
                    # Interpolate k from cv
                    # Ensure cv is within bounds of table
                    cv_clipped = np.clip(cv, _cv_values[0], _cv_values[-1])
                    _k_est = np.interp(cv_clipped, _cv_values, _k_values)
                    
                    # Calculate lambda
                    gamma_term = gamma(1 + 1/_k_est)
                    if gamma_term > 0:
                        _lambda_est = mu / gamma_term
                    else:
                        _lambda_est = mu # Fallback to exponential approx?
                        
                _last_fit_len = len(_items_seen)
            except Exception:
                # Fallback to defaults if fitting fails
                pass

    # Calculate scores
    # We want to maximize r * S(r)
    # S(r) = exp(-(r/lambda)^k)
    
    remaining = bins - item
    scores = np.full(len(bins), -1e9)
    
    valid_mask = remaining >= 0
    if np.any(valid_mask):
        r = remaining[valid_mask]
        
        # Calculate survival function S(r)
        # Handle potential numerical issues
        if _lambda_est > 0 and _k_est > 0:
            term = (r / _lambda_est) ** _k_est
            # Avoid overflow in exp
            term = np.clip(term, 0, 100) 
            s_r = np.exp(-term)
            scores[valid_mask] = r * s_r
        else:
            # Fallback to Best Fit if params are bad
            scores[valid_mask] = -r # Maximize -r is minimize r
            
    # Deadline guard
    if time.time() - start_time > 110.0:
        # Fallback to simple Best Fit to ensure return
        return -(bins - item)
        
    return scores
