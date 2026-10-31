import math
import time
import zlib

# Global deadline tracking
_start_time = None
_DEADLINE_FRACTION = 0.95  # Use 95% of time budget

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Primary: Prioritize vectors with all 3 distinct symbols.
    2. Secondary: Maximize uniformity of symbol distribution.
       Reward vectors where each symbol appears ~n/3 times, using a
       smooth quadratic penalty for deviation from uniform.
    3. Tertiary: Maximize Shannon Entropy as a complementary spread measure.
    4. Quaternary: Robust hash for deterministic tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Check deadline
    if time.time() - _start_time > 300 * _DEADLINE_FRACTION:
        return -1e10

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
    
    # Distinct count (3 is best, 2 is middle, 1 is worst)
    counts = [c0, c1, c2]
    distinct_count = sum(1 for c in counts if c > 0)
    
    # Uniformity score: reward being close to n/3 for each symbol
    # Use negative squared deviation from uniform as a smooth surrogate
    uniform_target = n / 3.0
    uniformity_penalty = 0.0
    for c in counts:
        diff = c - uniform_target
        uniformity_penalty += diff * diff
    # Normalize and invert: lower penalty is better
    uniformity_score = -uniformity_penalty / (n * n)
    
    # Shannon Entropy
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            entropy -= p * math.log(p)
    
    # Pairwise diversity: fraction of coordinate pairs with different values
    total_pairs = n * (n - 1) // 2
    if total_pairs > 0:
        same_pairs = (c0 * (c0 - 1) // 2) + (c1 * (c1 - 1) // 2) + (c2 * (c2 - 1) // 2)
        diverse_pairs = total_pairs - same_pairs
        diverse_ratio = diverse_pairs / total_pairs
    else:
        diverse_ratio = 0.0
    
    # Primary score: distinct count
    distinct_score = distinct_count * 10000.0
    
    # Secondary score: Higher uniformity (less penalty) is better
    uniformity_scaled = uniformity_score * 2000.0
    
    # Tertiary score: Higher entropy is better
    entropy_score = entropy * 500.0
    
    # Quaternary score: Higher pairwise diversity is better
    diverse_score = diverse_ratio * 300.0
    
    # Quinary: Robust hash
    h = 0
    base = 31
    for i, val in enumerate(vector):
        h = (h * base + val + i * 13) % (10**9 + 7)
    
    hash_score = h / (10**9 + 7)
    
    return float(distinct_score + uniformity_scaled + entropy_score + diverse_score + hash_score)
