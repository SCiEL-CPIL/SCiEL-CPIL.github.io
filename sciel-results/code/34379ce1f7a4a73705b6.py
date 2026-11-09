import time
import sys
import hashlib
import struct

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s kill

def priority(vector, n):
    """
    Prioritizes vectors based on their coordinate spread (pairwise variance).
    
    Structural heuristic: Vectors with high pairwise spread (sum of squared
    differences between all coordinate pairs) are less likely to participate
    in 3-term arithmetic progressions. This is a fundamentally different
    signal from global digit balance.
    """
    # Compute sum of squared differences between all pairs of coordinates
    # This is O(n^2) but n is typically small enough for this to be fast
    spread = 0.0
    for i in range(n):
        vi = vector[i]
        for j in range(i + 1, n):
            diff = vi - vector[j]
            spread += diff * diff
    
    # Normalize by n^2 to get a per-pair average
    if n > 1:
        spread /= (n * (n - 1))
    else:
        spread = 0.0
    
    # Add a small deterministic hash to break ties
    h = 2166136261
    for i, val in enumerate(vector):
        h = (h ^ (val + i)) * 16777619
        h &= 0xFFFFFFFF
    norm_h = (h % 10000000) / 10000000.0
    
    # Higher spread is better (more structurally complex vectors)
    return spread + norm_h * 0.001
