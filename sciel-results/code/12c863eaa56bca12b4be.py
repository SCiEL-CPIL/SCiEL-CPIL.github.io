"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import hashlib
import time

def priority(vector, n):
    """
    Priority function for cap set greedy insertion.
    
    Uses a combination of:
    1. A deterministic hash-based randomization to break lexicographic clustering
    2. A structural bias preferring vectors with more 0s (lower sum)
    
    Higher priority = inserted earlier.
    """
    # Convert vector to a string for hashing
    vec_str = ''.join(str(v) for v in vector)
    # Use MD5 hash for deterministic pseudo-random priority
    h = int(hashlib.md5(vec_str.encode()).hexdigest(), 16)
    random_part = (h % 1000000) / 1000000.0  # [0, 1)
    
    # Structural bias: prefer vectors with fewer 1s (they may create fewer APs)
    num_ones = sum(1 for v in vector if v == 1)
    num_zeros = sum(1 for v in vector if v == 0)
    
    # Higher priority for more zeros, fewer ones
    structural = 0.3 * num_zeros - 0.3 * num_ones
    
    # Combine: random part dominates, structural is a small bias
    return random_part + structural
# EVOLVE-BLOCK-END
