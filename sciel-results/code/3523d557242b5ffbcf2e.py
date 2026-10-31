import numpy as np
import time

_START_TIME = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 2.0

# Weibull parameters for item size distribution
# Typical online bin packing with capacity 100 often uses Weibull with
# shape k ~ 1-2 and scale lambda ~ 20-30 (mean ~ 20-30)
# We'll use k=1.5, lambda=20 as a reasonable default
WEIBULL_K = 1.5
WEIBULL_LAMBDA = 20.0

# Polarization strength: encourages bins to become either full or empty
# Higher values push more aggressively toward extreme remainders
POLARIZATION = 1.5

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline check
    if time.time() > DEADLINE - SAFETY_MARGIN:
        # Fallback to simple Best Fit
        feasible = bins >= item
        scores = np.full_like(bins, -np.inf, dtype=float)
        if np.any(feasible):
            rem = bins[feasible] - item
            scores[feasible] = -rem
        return scores

    C = 100.0
    n_bins = len(bins)
    
    # Identify feasible bins
    feasible = bins >= item
    if not np.any(feasible):
        return np.full_like(bins, -np.inf, dtype=float)
    
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Hybrid Strategy:
    # For large items (> C/2), use Best Fit to preserve larger bins for other large items.
    # For small items (<= C/2), use First Fit to consolidate items into earlier bins,
    # potentially closing bins sooner and reducing the number of open bins.
    
    # Use Weibull CDF-based scoring
    # The idea: for each feasible bin, compute the remaining space after placing the item
    # Then score based on how likely that remaining space is to be filled by future items
    # Using Weibull CDF: P(X <= rem) where X ~ Weibull(k, lambda)
    # Higher CDF value means the remaining space is more likely to be filled
    # We want to maximize this, so we use it directly as the score
    
    rem = bins[feasible] - item
    
    # Weibull CDF: F(x) = 1 - exp(-(x/lambda)^k)
    # For x >= 0
    # We want to favor bins where the remaining space is likely to be filled
    # So higher F(rem) is better
    
    # However, we also need to consider that very small remainders are easy to fill
    # but very large remainders might not be fillable
    # The Weibull CDF naturally handles this: it increases from 0 to 1
    
    weibull_cdf = 1.0 - np.exp(-np.power(rem / WEIBULL_LAMBDA, WEIBULL_K))
    
    # Score: higher is better, so we use the CDF value directly
    # But we need to be careful: if rem is very small, CDF is small
    # if rem is large, CDF approaches 1
    # This means we prefer bins where the remaining space is "typical" for future items
    
    # Actually, let's think about this more carefully:
    # We want to place the item in a bin such that the remaining space is most likely
    # to be filled by future items. So we want to maximize P(future_item <= rem)
    # which is exactly the Weibull CDF at rem
    
    scores[feasible] = weibull_cdf
    
    return scores
