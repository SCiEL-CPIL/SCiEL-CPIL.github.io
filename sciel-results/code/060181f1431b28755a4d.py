import time
import random

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # Seconds

# We will use a static priority function based on:
# 1. Sum of squares of coordinates (non-linear measure)
# 2. A deterministic hash for tie-breaking
#
# Rationale:
# - Vectors with more 1s and 2s have higher sum of squares.
# - In GF(3), lines are defined by a, a+d, a+2d.
# - By prioritizing vectors with higher "energy", we might break the symmetry
#   that leads to dense clusters of low-energy vectors forming lines.
# - This is a structural change from the "balance" (deviation from mean) to
#   "energy" (sum of squares), which is a different metric family.

def _hash_vector(vec, n):
    """Deterministic hash for tie-breaking."""
    # Use a simple polynomial hash
    h = 0
    base = 31
    for v in vec:
        h = (h * base + v) % (10**9 + 7)
    return h / (10**9 + 7)

def priority(vector, n):
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Fallback to a very fast simple score
        # Just sum of squares
        return -sum(v * v for v in vector)
        
    # Calculate sum of squares
    sum_sq = 0
    for v in vector:
        sum_sq += v * v
    
    # The strategy says "favoring vectors with more 1s and 2s over 0s"
    # So higher sum_sq should be higher priority.
    # But we also need to break ties deterministically.
    
    # Let's combine them.
    # We want to prioritize higher sum_sq.
    # However, if we just use sum_sq, many vectors will have the same sum_sq.
    # We need a tie-breaker that is consistent but effectively randomizes the order
    # within the same sum_sq bucket to avoid systematic biases.
    
    # Let's use a hash that depends on the vector and n.
    h = _hash_vector(vector, n)
    
    # Scale the hash to a small range so it doesn't dominate the sum_sq
    # sum_sq ranges from 0 to 4*n
    # We want the primary sort key to be sum_sq, and secondary to be hash.
    
    # Return a float where higher is better.
    # primary = sum_sq
    # secondary = hash
    # combined = primary + secondary * small_factor
    
    # To ensure distinct values for sorting, we can mix them.
    # But simply summing might cause collisions if hash is too fine-grained.
    # Let's just return sum_sq + hash.
    
    return float(sum_sq) + h

# Note: The original program had a complex pre-computation logic.
# This new strategy is a simple static function, which should be very fast
# and avoid any timeout issues.
# The "structural change" is moving from a balance-based (deviation from mean)
# to an energy-based (sum of squares) metric, which breaks the symmetry differently.
