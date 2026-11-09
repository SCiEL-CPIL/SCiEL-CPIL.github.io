import time
import math

_T0 = time.time()
_DEADLINE = _T0 + 280.0

def priority(vector, n):
    """
    Priority for a vector in {0,1,2}^n for greedy cap-set construction.
    
    Strategy:
    1. Structural Balance: Vectors with symbol counts close to n/3 are preferred.
    2. Midpoint Penalty: Vectors with many 1s are penalized as they are frequent midpoints.
    3. Deterministic Hash: A high-quality hash breaks symmetries and prevents 
       the greedy algorithm from clustering in specific structural regions, 
       allowing for a more diverse and larger cap set.
    """
    if time.time() > _DEADLINE:
        return 0.0

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

    # 1. Structural Balance Score
    # Target is n/3 of each symbol. Use a smooth penalty that rewards
    # balanced compositions more strongly than the raw squared deviation,
    # since the greedy pass benefits from routing through the high-density
    # centre of the affine cube first.
    target = n / 3.0
    # Normalize deviation to [0,1] scale for stable weighting
    dev0 = abs(c0 - target) / max(n, 1)
    dev1 = abs(c1 - target) / max(n, 1)
    dev2 = abs(c2 - target) / max(n, 1)
    # Smooth min-like penalty: penalize the max deviation more than the sum
    # to prefer vectors that are uniformly balanced across all three symbols
    max_dev = max(dev0, dev1, dev2)
    sum_dev = dev0 + dev1 + dev2
    penalty_balance = 2.0 * max_dev + 0.5 * sum_dev
    
    # 2. Midpoint Penalty
    # 1s are the midpoints of 0 and 2. Having many 1s makes the vector a 
    # likely midpoint for many pairs, which is bad for cap sets.
    # Increase weight to more strongly discourage midpoint-rich vectors.
    penalty_midpoint = 0.3 * c1 / max(n, 1)
    
    # Combined structural penalty
    structural_score = -penalty_balance - penalty_midpoint

    # 3. Deterministic Hash for Tie-breaking and Diversity
    # We create a unique integer representation of the vector.
    # Since n is small (typically < 20 for these tasks), we can pack it.
    # Base 3 encoding is natural.
    val = 0
    for i, v in enumerate(vector):
        val = val * 3 + v
        
    # Mix the integer to get a pseudo-random distribution
    # Using a simple but effective bit-mixing function
    val = (val ^ (val >> 16)) * 0x45d9f3b
    val = (val ^ (val >> 16)) * 0x45d9f3b
    val = val ^ (val >> 16)
    
    # Convert to float in [-1, 1]
    # Normalize by a large power of 2 to avoid overflow issues in conversion
    hash_float = (val % 100000) / 50000.0 - 1.0

    # Combine scores
    # The hash term is small enough to break ties but not override structural logic.
    # We scale it slightly to ensure it has impact on close calls.
    final_score = structural_score + hash_float * 0.3
    
    return final_score
