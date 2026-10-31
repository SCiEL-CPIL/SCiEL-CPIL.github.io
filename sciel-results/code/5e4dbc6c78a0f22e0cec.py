import math
import time

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 2.0

# Cache for priority values
_priority_cache = {}

def _time_check():
    return (time.time() - _START_TIME) > (_TIME_LIMIT - _SAFETY_MARGIN)

def priority(vector, n):
    key = (vector, n)
    if key in _priority_cache:
        return _priority_cache[key]
    
    if _time_check():
        # Fallback to a simple deterministic score if we're running out of time
        result = 0.0
        _priority_cache[key] = result
        return result
    
    # Count occurrences of 0, 1, 2
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    
    # Ideal count per symbol for balanced vector
    ideal = n / 3.0
    
    # Deviation from ideal balance (L1 distance)
    deviation = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    
    # We want to MINIMIZE deviation, so negate for "higher is better"
    # Scale to dominate tie-breakers
    balance_score = -deviation * 10000.0
    
    # Deterministic polynomial hash for tie-breaking
    h = 0
    base = 31
    for val in vector:
        h = (h * base + val) % (10**9 + 7)
    
    # Map hash to small range [0, 1) for tie-breaking
    hash_score = h / (10**9 + 7)
    
    result = float(balance_score + hash_score)
    
    _priority_cache[key] = result
    return result
