import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Estimate the mode of the Weibull distribution of item sizes.
    # Based on typical online bin packing benchmarks with capacity 100,
    # item sizes often follow Weibull(k=1.5, lambda=11) or similar,
    # yielding a mode around 5-6.
    mode = 6.0
    
    remaining = bins - item
    scores = np.full(bins.shape, -np.inf, dtype=float)
    
    valid = remaining >= 0
    if np.any(valid):
        # Score is negative squared deviation from the mode.
        # This penalizes remainders that are too small (dead space) 
        # and too large (wasted capacity), preferring remainders near the mode.
        scores[valid] = -((remaining[valid] - mode) ** 2)
        
    return scores
