import math
import time

# Global deadline tracking
_start_time = None
_DEADLINE_FRACTION = 0.95  # Use 95% of time budget

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    A cap set has no three distinct vectors a, b, c with a + b + c ≡ 0 (mod 3).
    
    Strategy: Prioritize vectors with symbol counts close to uniform (n/3 each),
    measured by chi-squared deviation. This makes vectors "generic" so their
    AP partners are also generic, reducing collisions with already-added vectors.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Count symbols
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
    
    # Chi-squared deviation from uniform distribution
    ideal = n / 3.0
    if ideal > 0:
        chi2 = ((c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2) / ideal
    else:
        chi2 = 0.0
    
    # Primary score: lower chi2 = higher priority
    # Scale to make it dominant
    balance_score = -chi2 * 1000.0
    
    # Secondary: structural hash for deterministic tie-breaking
    h = 0
    base = 31
    for i, val in enumerate(vector):
        h = (h * base + val + i * 7) % (10**9 + 7)
    
    hash_score = h / (10**9 + 7)
    
    return float(balance_score + hash_score)
