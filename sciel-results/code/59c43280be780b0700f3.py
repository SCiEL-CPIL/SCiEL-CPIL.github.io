"""Cap set greedy construction with structure-aware static priority."""
import time
import math
import random

_START_TIME = time.time()
TIME_LIMIT = 280  # seconds

def _strong_hash(vector, n):
    """
    Strong deterministic hash for tie-breaking.
    Uses a multi-round mixing approach to produce a well-distributed
    64-bit-like integer that breaks symmetries effectively.
    """
    # Round 1: FNV-1a style
    h1 = 14695981039346656037  # FNV offset basis (64-bit)
    for i in range(n):
        h1 ^= vector[i]
        h1 = (h1 * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    
    # Round 2: Position-weighted sum with prime multipliers
    h2 = 0
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
              59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
              127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
              191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251,
              257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
              331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397,
              401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
              467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557,
              563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619,
              631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701,
              709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787,
              797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863,
              877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953,
              967, 971, 977, 983, 991, 997]
    for i in range(n):
        h2 = (h2 + vector[i] * primes[i % len(primes)]) & 0xFFFFFFFFFFFFFFFF
    
    # Round 3: Adjacent-pair interaction hash
    h3 = 0
    for i in range(n - 1):
        pair_val = vector[i] * 3 + vector[i + 1]
        h3 = (h3 * 31 + pair_val) & 0xFFFFFFFFFFFFFFFF
    
    # Combine all three hashes into a single 64-bit value
    combined = (h1 ^ (h2 << 1) ^ (h3 >> 1)) & 0xFFFFFFFFFFFFFFFF
    return combined

def priority(vector, n):
    """
    Static priority function for greedy cap set construction.
    
    Strategy:
    1. Favor vectors with balanced digit distribution (0, 1, 2 counts close to n/3).
    2. Favor vectors with low "self-correlation" (fewer pairs of equal coordinates),
       as these are more 'generic' and less likely to form structured APs with
       other generic vectors.
    3. Use a strong multi-round hash for deterministic tie-breaking to break
       symmetries that cause early AP formation.
    """
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    target = n / 3.0
    
    # 1. Balance Penalty: Minimize deviation from uniform distribution
    #    Use a smooth surrogate: soft-max of squared deviations to better
    #    distinguish near-ties while maintaining the tier structure.
    dev0 = (c0 - target) ** 2
    dev1 = (c1 - target) ** 2
    dev2 = (c2 - target) ** 2
    
    # Soft-max with temperature for smoothness
    temp = 1.0
    max_dev = max(dev0, dev1, dev2)
    if max_dev > 0:
        exp0 = math.exp(dev0 / temp - max_dev / temp)
        exp1 = math.exp(dev1 / temp - max_dev / temp)
        exp2 = math.exp(dev2 / temp - max_dev / temp)
        # Soft-max gives a smooth measure of the dominant deviation
        soft_max = max_dev + temp * math.log(exp0 + exp1 + exp2)
    else:
        soft_max = 0.0
    
    # Also keep the raw sum for additional granularity
    balance_penalty = dev0 + dev1 + dev2
    
    # 2. Structure Penalty: Minimize number of equal coordinate pairs
    def comb2(k):
        return k * (k - 1) // 2
    
    equal_pairs = comb2(c0) + comb2(c1) + comb2(c2)
    
    # 3. Sum Deviation Penalty
    s = c1 + 2 * c2
    sum_penalty = (s - n) ** 2
    
    # 4. Pairwise Variance: Measure spread across coordinates
    if n > 0:
        mean = s / n
        mean_sq = (c1 + 4 * c2) / n
        variance = mean_sq - mean * mean
    else:
        variance = 0.0
    
    # 5. Strong tie-breaker hash
    h = _strong_hash(vector, n)
    
    # Combine scores
    # We want to MAXIMIZE the return value.
    # 
    # Weight hierarchy:
    # - Balance penalty dominates to establish clear tiers
    # - Soft-max provides smooth distinction within near-ties
    # - Variance provides meaningful tie-breaking within balance tiers
    # - Structure penalty is tertiary
    # - Sum penalty is quaternary
    # - Strong hash provides robust symmetry-breaking (1e-8 scale)
    
    score = -balance_penalty * 1000.0
    score -= soft_max * 50.0  # Smooth penalty for dominant deviation
    score += variance * 100.0  # Favor more spread (higher variance)
    score -= equal_pairs * 50.0
    score -= sum_penalty * 10.0
    score += h * 1e-8  # Stronger hash perturbation for symmetry-breaking
    
    return score
