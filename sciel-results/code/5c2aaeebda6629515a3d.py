import math
from functools import lru_cache

@lru_cache(maxsize=None)
def _entropy(c0, c1, c2, n):
    """Compute Shannon entropy for given counts."""
    h = 0.0
    if c0 > 0:
        p0 = c0 / n
        h -= p0 * math.log2(p0)
    if c1 > 0:
        p1 = c1 / n
        h -= p1 * math.log2(p1)
    if c2 > 0:
        p2 = c2 / n
        h -= p2 * math.log2(p2)
    return h

def priority(vector, n):
    # Count 0s, 1s, 2s
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = n - c0 - c1
    
    # Base score: Shannon entropy
    h = _entropy(c0, c1, c2, n)
    
    # Add a small perturbation based on the vector's structure to break ties
    # Use a hash-based deterministic perturbation that's small enough not to
    # dominate the entropy but large enough to break ties consistently
    # The perturbation should be in [0, 1e-6] so it doesn't affect ordering of
    # vectors with significantly different entropies
    
    # Use sum of positions weighted by value as a simple deterministic feature
    # This gives different vectors with same (c0,c1,c2) different priorities
    s = sum(i * v for i, v in enumerate(vector))
    # Normalize to [0, 1] and scale to a very small range
    max_s = n * (n - 1)  # max possible sum when all 2s are at end
    if max_s > 0:
        perturbation = (s / max_s) * 1e-6
    else:
        perturbation = 0.0
    
    return h + perturbation
