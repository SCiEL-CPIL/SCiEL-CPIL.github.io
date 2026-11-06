def priority(vector, n):
    """
    Prioritizes vectors based on the number of zero entries (Hamming weight complement),
    with a secondary structural feature capturing balance between 1s and 2s.
    Hypothesis: Vectors with more zeros are 'sparser' and less likely to form
    3-term arithmetic progressions. Among vectors with equal zero counts, those
    with more balanced 1/2 distributions may have fewer interactions.
    """
    # 1. Count zeros, ones, and twos
    zero_count = 0
    one_count = 0
    two_count = 0
    for v in vector:
        if v == 0:
            zero_count += 1
        elif v == 1:
            one_count += 1
        else:  # v == 2
            two_count += 1
    
    # 2. Base score: Higher zero count is primary.
    base_score = float(zero_count)
    
    # 3. Secondary: balance of non-zeros. 
    # In F_3, a 3-AP condition involves x+z=2y mod 3.
    # For fixed zero count, vectors with similar 1/2 ratios may form APs more easily.
    # We reward vectors where |one_count - two_count| is small (balanced),
    # but scale this down so it only breaks ties within zero-count classes.
    non_zero = one_count + two_count
    if non_zero > 0:
        imbalance = abs(one_count - two_count) / non_zero
        # Balance bonus: 1.0 when perfectly balanced, 0.0 when all one type
        balance_bonus = 1.0 - imbalance
    else:
        balance_bonus = 1.0  # All zeros, perfectly balanced by default
    
    # Add balance as a small secondary term (scaled to not override zero count)
    base_score += 0.1 * balance_bonus
    
    # 4. Generate a pseudo-random tie-breaker
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
    
    # 5. Combine
    return base_score + random_frac
