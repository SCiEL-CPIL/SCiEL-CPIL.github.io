"""Cap set greedy construction with structure-aware static priority."""
import time
import math
import random

_START_TIME = time.time()
TIME_LIMIT = 280  # seconds

def priority(vector, n):
    """
    Static priority function for greedy cap set construction.
    
    Strategy:
    1. Favor vectors with balanced digit distribution (0, 1, 2 counts close to n/3).
    2. Favor vectors with low "self-correlation" (fewer pairs of equal coordinates),
       as these are more 'generic' and less likely to form structured APs with
       other generic vectors.
    3. Use a robust hash for deterministic tie-breaking to avoid symmetry traps.
    """
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    target = n / 3.0
    
    # 1. Balance Penalty: Minimize deviation from uniform distribution
    balance_penalty = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    
    # 2. Structure Penalty: Minimize number of equal coordinate pairs
    #    High number of equal pairs indicates the vector lies on a "diagonal"
    #    or has strong symmetry, which can be detrimental in cap set context.
    def comb2(k):
        return k * (k - 1) // 2
    
    equal_pairs = comb2(c0) + comb2(c1) + comb2(c2)
    
    # 3. Sum Deviation Penalty
    #    Prefer vectors where sum of digits is close to n (average value 1)
    s = c1 + 2 * c2
    sum_penalty = (s - n) ** 2
    
    # 4. Pairwise Variance: Measure spread across coordinates
    #    For vectors with the same balance penalty, those with more
    #    variation in digit values across positions are more "generic"
    #    and less likely to form 3-term APs with similar vectors.
    #    Variance = E[X^2] - (E[X])^2 where X is the digit value
    #    E[X] = s/n, E[X^2] = (0*c0 + 1*c1 + 4*c2)/n
    if n > 0:
        mean = s / n
        mean_sq = (c1 + 4 * c2) / n
        variance = mean_sq - mean * mean
    else:
        variance = 0.0
    
    # 5. Tie-breaker: Deterministic pseudo-random hash
    #    This ensures that among vectors with similar structural stats,
    #    the selection order is stable but not easily exploited by symmetry.
    h = 0
    for i in range(n):
        h = (h * 31 + vector[i]) % (10**9 + 7)
    
    # Combine scores
    # We want to MAXIMIZE the return value.
    # Penalties are subtracted.
    # 
    # Weight hierarchy:
    # - Balance penalty dominates (1000x scale) to establish clear tiers
    # - Variance provides meaningful tie-breaking within balance tiers (100x)
    # - Structure penalty is tertiary (50x)
    # - Sum penalty is quaternary (10x)
    # - Hash is for exact ties (1e-9)
    
    score = -balance_penalty * 1000.0
    score += variance * 100.0  # Favor more spread (higher variance)
    score -= equal_pairs * 50.0
    score -= sum_penalty * 10.0
    score += h * 1e-9  # Tiny perturbation for tie-breaking
    
    return score
