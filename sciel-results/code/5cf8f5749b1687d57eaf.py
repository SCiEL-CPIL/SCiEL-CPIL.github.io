import time

_start_time = time.time()
_TIME_LIMIT = 280.0  # Safety margin before 330s kill

def _is_time_up():
    """Check if we're approaching the time limit."""
    return (time.time() - _start_time) > _TIME_LIMIT

def priority(vector, n):
    """
    Prioritizes vectors using a balanced symbol distribution heuristic with
    structured tie-breaking to encourage structural diversity in the cap set.
    
    Key insight: Vectors with balanced symbol counts (close to n/3 each) are
    more structurally diverse and less likely to form 3-term arithmetic
    progressions, leading to larger cap sets.
    """
    # Quick time check - if we're close to limit, use fast path
    if _is_time_up():
        # Fast deterministic tie-breaker
        val = 0
        for v in vector:
            val = val * 3 + v
        return float(val)
    
    # Count occurrences of each symbol
    count_0 = 0
    count_1 = 0
    count_2 = 0
    for v in vector:
        if v == 0:
            count_0 += 1
        elif v == 1:
            count_1 += 1
        else:
            count_2 += 1
    
    # Balance score: minimize deviation from uniform distribution (n/3 each)
    # Use sum of squared deviations as a smooth surrogate for min-max balance
    expected = n / 3.0
    deviation = ((count_0 - expected) ** 2 + 
                 (count_1 - expected) ** 2 + 
                 (count_2 - expected) ** 2)
    
    # Lower deviation is better, so we negate for priority
    balance_score = -deviation
    
    # Secondary: prefer vectors with all three symbols present (structural diversity)
    distinct_count = (1 if count_0 > 0 else 0) + (1 if count_1 > 0 else 0) + (1 if count_2 > 0 else 0)
    diversity_bonus = float(distinct_count) * 0.1
    
    # Tertiary: slight preference for higher Hamming weight (more 1s and 2s)
    hamming_weight = count_1 + count_2
    weight_bonus = float(hamming_weight) * 0.01
    
    # Combined base score
    base_score = balance_score + diversity_bonus + weight_bonus
    
    # Generate high-quality pseudo-random tie-breaker using SplitMix64
    val = 0
    for v in vector:
        val = val * 3 + v
        
    # Apply SplitMix64 mixing for good distribution
    val = (val + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
    val = (val ^ (val >> 30)) * 0xbf58476d1ce4e5b9 & 0xFFFFFFFFFFFFFFFF
    val = (val ^ (val >> 27)) * 0x94d049bb133111eb & 0xFFFFFFFFFFFFFFFF
    val = val ^ (val >> 31)
    
    # Convert to fraction [0, 1)
    random_frac = (val >> 11) / (1 << 53)
    
    # Combine: balance dominates, with small structural bonuses and random tie-break
    return base_score + diversity_bonus * 0.01 + random_frac
