import numpy as np
import time

# Module-level deadline guard state
_START = None
_DEADLINE_S = 300.0  # safety margin well under the 330s kill

def _elapsed():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START

def _deadline_ok():
    return _elapsed() < _DEADLINE_S

def acquisition(mu, sigma, f_best):
    global _START
    if _START is None:
        _START = time.time()

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety: ensure finite inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e9, neginf=-1e9)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    sigma = np.maximum(sigma, 1e-12)

    # Deadline guard: if we are close to the limit, return a safe fallback
    if not _deadline_ok():
        # Fallback: pure exploitation (lowest mu)
        return np.nan_to_num(-mu, nan=0.0, posinf=1e6, neginf=-1e6)

    # Structural Change: Capped UCB
    # Instead of using raw sigma, we cap the exploration term to prevent
    # regions with extremely high uncertainty (likely far from data/noise)
    # from dominating the acquisition score. This focuses exploration on
    # "plausibly" uncertain regions.
    
    # Determine a cap for sigma. 
    # Using the 95th percentile of the current sigma distribution.
    # This adapts to the current state of the GP.
    sigma_cap = np.percentile(sigma, 95)
    
    # Ensure cap is at least the minimum sigma to avoid division by zero or negative caps
    if sigma_cap < 1e-9:
        sigma_cap = 1e-9
        
    sigma_capped = np.minimum(sigma, sigma_cap)
    
    # UCB with capped sigma
    # Beta remains 1.5 as it was the best fixed constant in the parent
    beta = 1.5
    
    scores = -mu + beta * sigma_capped
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
