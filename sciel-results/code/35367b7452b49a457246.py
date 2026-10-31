import numpy as np
import time
from scipy.special import erf
from scipy.stats import weibull_min

# Global state
_start_time = None
_sum_items = 0.0
_count_items = 0
_WEIBULL_K = 2.0
# Gamma(1 + 1/2) = Gamma(1.5) = sqrt(pi)/2
_GAMMA_1_5 = np.sqrt(np.pi) / 2.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _start_time, _sum_items, _count_items
    
    # Initialize timing if not started
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: 150s limit, keep 5s margin
    elapsed = time.time() - _start_time
    if elapsed > 145:
        # Fallback to simple Best Fit to ensure completion
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores

    # Update statistics
    _sum_items += item
    _count_items += 1
    
    # Estimate mean
    if _count_items > 0:
        mean_item = _sum_items / _count_items
    else:
        mean_item = 15.0 # Initial guess
        
    # Estimate lambda for Weibull(k=2)
    # Mean = lambda * Gamma(1 + 1/k) => lambda = Mean / Gamma(1.5)
    lambda_est = mean_item / _GAMMA_1_5
    
    # Safety check: if lambda is too small or zero, use a default
    if lambda_est < 1e-6:
        lambda_est = 15.0

    # Calculate remaining capacity
    remaining = bins - item
    
    # Identify valid bins (enough capacity)
    valid = remaining >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid):
        return scores
        
    valid_indices = np.where(valid)[0]
    new_rems = remaining[valid]
    
    # Calculate Expected Waste W(r) = r - E[min(X, r)]
    # E[min(X, r)] = integral_0^r S(t) dt
    # For Weibull(k=2, lambda): S(t) = exp(-(t/lambda)^2)
    # integral_0^r exp(-(t/lambda)^2) dt = (sqrt(pi)*lambda/2) * erf(r/lambda)
    
    # Vectorized calculation
    erf_term = erf(new_rems / lambda_est)
    expected_min = (np.sqrt(np.pi) * lambda_est / 2.0) * erf_term
    waste = new_rems - expected_min
    
    # Score is negative waste
    # We want to minimize waste, so maximize -waste
    scores_valid = -waste
    
    # Tie-breaker: prefer lower index
    scores_valid -= np.arange(len(valid_indices)) * 1e-9
    
    scores[valid] = scores_valid
    
    return scores
