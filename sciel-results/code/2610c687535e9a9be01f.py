import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before 120s limit
    
    remaining = bins - item
    feasible = remaining >= 0
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(feasible):
        return scores
    
    rem = remaining[feasible]
    
    # Base score: prefer tight fits (best-fit decreasing style)
    base = -rem.copy()
    
    # Smooth quadratic penalty for very tight fits
    # Capacity is 100, threshold at 8.0 (8% remaining)
    threshold = 8.0
    tight_mask = rem < threshold
    if np.any(tight_mask):
        excess = threshold - rem[tight_mask]
        # Quadratic penalty: smooth and stronger for very tight fits
        base[tight_mask] -= 0.3 * excess * excess
    
    # Secondary: prefer bins with higher utilization (consolidate items)
    # This helps reduce the number of bins opened
    used = bins[feasible] - rem  # This is just bins[feasible], but let's be explicit
    # Normalize utilization: higher is better for consolidation
    # Weight lightly so it doesn't override the tight-fit preference
    base += 0.02 * used
    
    scores[feasible] = base
    
    # Time check - if we're running long, just return what we have
    if time.time() > deadline:
        return scores
    
    return scores
