"""Cap set construction via greedy selection with hybrid priorities."""
import time

# EVOLVE-BLOCK-START

def _fnv1a_hash(vector, seed=0):
    """Fast non-linear hash using FNV-1a variant with avalanche mixing."""
    h = 1469598103934665603  # FNV offset basis
    for v in vector:
        h ^= v
        h *= 1099511628211   # FNV prime
    h ^= seed
    # 64-bit finalizer (MurmurHash3-style)
    h ^= (h >> 33)
    h *= 0xff51afd7ed558ccd
    h ^= (h >> 33)
    h *= 0xc4ceb9fe1a85ec53
    h ^= (h >> 33)
    # Convert to float in [0, 1)
    return (h & 0xFFFFFFFFFFFFFFFF) / (1 << 64)

def _structural_priority(vector, n):
    """
    Structural score based on weight.
    Favors vectors with weight close to n (the mean weight).
    This targets the 'center' of the hypercube, which often has
    good packing properties.
    """
    w = sum(vector)
    # Deviation from mean weight n
    dev = abs(w - n)
    # Normalize to [0, 1]
    # Max deviation is n
    score = 1.0 - (dev / n)
    return score

def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    Combines structural weight preference with non-linear hash diversity.
    Higher priority = selected earlier.
    """
    s = _structural_priority(vector, n)
    h1 = _fnv1a_hash(vector, seed=1)
    h2 = _fnv1a_hash(vector, seed=2)
    
    # Blend: 20% structure, 80% hash
    # Strong hash component ensures diverse ordering, breaking symmetries
    # that might lead to early saturation.
    return 0.2 * s + 0.6 * h1 + 0.2 * h2

# EVOLVE-BLOCK-END
