"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 120.0
_start_time = time.time()

# We assume a standard Weibull distribution for the items.
# Common defaults in bin packing benchmarks often use k=1.0 (Exponential) or k=2.0.
# Given the "Weibull sequences" and capacity 100, let's assume a distribution
# that is reasonable for this scale. 
# If k=1.0 (Exponential), the mean is lambda. 
# If items are roughly in [0, 100], a scale of ~20-30 might be typical.
# However, without specific parameters, we can use a heuristic that is robust.
# The "effective capacity" idea: 
# We want to place an item in a bin such that the remaining space R is "valuable".
# R is valuable if it is likely to be filled by a future item.
# If we assume the next item has expected size E[X], then the "net" remaining 
# capacity after the next item is R - E[X].
# We want to maximize the probability that R >= X_next.
# This is equivalent to maximizing F(R) where F is the CDF.
# Alternatively, we can approximate the "expected waste" of leaving gap R.
# If R < E[X], it's likely to be wasted. If R > E[X], it's likely to be used.
# Let's try scoring based on the "effective capacity" = R - E[X].
# If we maximize this, we are effectively doing Worst-Fit but shifted by the expected item size.
# This encourages leaving a gap that is at least as large as the expected next item.

# Let's estimate E[X]. 
# If we don't know the distribution, we can't compute E[X] precisely.
# However, we can use a heuristic. 
# Let's assume the items are Weibull(k=1.0, scale=20). Then E[X] = 20.
# Or Weibull(k=2.0, scale=30). Then E[X] = 30 * Gamma(1.5) ≈ 27.
# Let's try a fixed E[X] = 25 as a reasonable middle ground for capacity 100.
# Or, better, let's use the item size itself as a proxy for the scale? No, that's noisy.
# Let's stick to a constant expected size derived from typical Weibull parameters.
# If the task uses a specific Weibull, it might be k=1.0, lambda=10? No, capacity 100.
# Let's try E[X] = 20.

E_X = 20.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Fallback to a simple strategy if time is running out
        return -(bins - item)
    
    remaining = bins - item
    feasible = remaining >= 0
    
    # Strategy: Maximize (Remaining - E[X])
    # This is a "Shifted Worst-Fit".
    # It prefers bins with large remaining space, but penalizes bins where the remaining space
    # is less than the expected size of the next item (implying high waste probability).
    # Actually, if R < E[X], R - E[X] is negative.
    # If R > E[X], R - E[X] is positive.
    # So we prefer bins where R > E[X].
    # Among those, we prefer larger R.
    # This is structurally different from pure Best-Fit (minimize R) and pure Worst-Fit (maximize R).
    # It introduces a "threshold" behavior.
    
    scores = np.where(feasible, remaining - E_X, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
