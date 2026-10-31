import math
from functools import lru_cache

@lru_cache(maxsize=None)
def _entropy_term(c0, c1, c2, n):
    """Compute Shannon entropy for counts (c0, c1, c2) in dimension n."""
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

@lru_cache(maxsize=None)
def _balance_term(c0, c1, c2, n):
    """
    Smooth surrogate for how 'balanced' the symbol counts are.
    Uses a soft-max of pairwise count differences to penalize imbalance.
    Returns a value in [0, 1] where 1 = perfectly balanced.
    """
    if n == 0:
        return 1.0
    # Pairwise absolute differences, normalized
    d01 = abs(c0 - c1) / n
    d02 = abs(c0 - c2) / n
    d12 = abs(c1 - c2) / n
    
    # Soft max via log-sum-exp for smoothness (beta controls sharpness)
    beta = 4.0
    max_d = max(d01, d02, d12)
    if max_d < 1e-15:
        return 1.0
    
    # exp(-beta * d) for each pair difference, then average
    # This gives a smooth measure that's high when all counts are close
    s = math.exp(-beta * d01) + math.exp(-beta * d02) + math.exp(-beta * d12)
    return s / 3.0

def priority(vector, n):
    # Count 0s, 1s, 2s
    # vector is a tuple of 0, 1, 2
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = n - c0 - c1
    
    # Primary: Shannon entropy (maximizes symbol diversity)
    h = _entropy_term(c0, c1, c2, n)
    
    # Secondary: balance term (smooth surrogate for even distribution)
    # Weighted small so entropy dominates but ties are broken by balance
    b = _balance_term(c0, c1, c2, n)
    
    # Combine: entropy is primary, balance breaks ties smoothly
    # Scale balance to be a small perturbation of entropy
    return h + 0.05 * b
