"""Rewritten program for cap set greedy priority."""
import time

def priority(vector, n):
    """
    Assign a priority to a vector in {0,1,2}^n for greedy cap set construction.
    
    Strategy: Use a combination of:
    1. Sum of coordinates (prefer balanced vectors)
    2. Number of distinct coordinates (prefer vectors with more distinct values)
    3. Product of non-zero coordinates (tie-breaker)
    4. Lexicographic order as final tie-breaker
    
    Higher priority means more likely to be added first.
    """
    # Time guard - return a deterministic priority quickly
    t0 = time.time()
    
    # Calculate features
    coord_sum = sum(vector)
    distinct_count = len(set(vector))
    
    # Prefer vectors with more distinct coordinates (harder to form lines)
    # and a sum closer to the middle of the range
    mid_sum = 1.5 * n  # average if all coordinates were 1.5
    
    # Distance from middle sum - prefer closer to middle
    sum_distance = abs(coord_sum - mid_sum)
    
    # Score: higher distinct count is better, smaller sum distance is better
    # Combine into a single priority
    priority_val = distinct_count * 100.0 - sum_distance * 10.0
    
    # Add a small deterministic tie-breaker based on lexicographic order
    # to ensure consistent ordering
    tie_break = 0.0
    for i, v in enumerate(vector):
        tie_break += v * (1.0 / (n + i + 1))
    
    priority_val += tie_break * 0.001
    
    return priority_val
