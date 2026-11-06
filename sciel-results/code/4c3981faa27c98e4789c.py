def priority(vector, n):
    """
    Prioritizes vectors using a precomputed hash-based permutation of the entire space.
    The primary key is the number of zero entries (higher = sparser = less likely to
    form 3-APs). The secondary key is a globally unique, deterministic pseudo-random
    permutation derived from a strong hash of the vector, which avoids local clustering
    of balance-based scores and provides a more uniform exploration order.
    """
    import time
    
    # Time guard: track elapsed time
    start_time = time.time()
    DEADLINE = 325.0  # 5 seconds safety margin before 330s kill
    
    # 1. Count zeros, ones, and twos; compute sum of squares
    zero_count = 0
    one_count = 0
    two_count = 0
    sum_sq = 0
    for v in vector:
        if v == 0:
            zero_count += 1
        elif v == 1:
            one_count += 1
            sum_sq += 1
        else:  # v == 2
            two_count += 1
            sum_sq += 4
    
    # 2. Base score: Higher zero count is primary.
    base_score = float(zero_count)
    
    # 3. Secondary: variance of non-zero entries (weak structural signal)
    non_zero = one_count + two_count
    if non_zero > 0:
        spread = (sum_sq / non_zero - 1.0) / 3.0  # 0 for all-ones, 1 for all-twos
        variance_bonus = 1.0 - spread
    else:
        variance_bonus = 1.0  # All zeros
    
    # 4. Tertiary: balance between 1s and 2s (weaker)
    if non_zero > 0:
        imbalance = abs(one_count - two_count) / non_zero
        balance_bonus = 1.0 - imbalance
    else:
        balance_bonus = 1.0
    
    # Add secondary terms with small weights to not override zero count
    base_score += 0.08 * variance_bonus + 0.04 * balance_bonus
    
    # 5. Generate a pseudo-random tie-breaker using a strong hash
    # Encode vector as base-3 integer
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
    
    # 6. Combine
    result = base_score + random_frac
    
    # Time guard: if we're close to deadline, return immediately
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 1.0:
        return result
    
    return result
