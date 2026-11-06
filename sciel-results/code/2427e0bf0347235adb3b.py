def priority(vector, n):
    # Structural score: prefer vectors with balanced coordinate distribution
    # and low "3-AP middle" risk. Use FNV-1a hash as tie-breaker.
    counts = [0, 0, 0]
    for x in vector:
        counts[x] += 1
    
    # Balance score: closer to uniform is better
    # Variance of counts (lower = more balanced)
    mean_c = n / 3.0
    var_c = sum((c - mean_c) ** 2 for c in counts) / 3.0
    balance_score = -var_c  # lower variance = higher score
    
    # 3-AP middle risk: a vector v is "middle" of a 3-AP (a, v, c) if
    # a + c = 2*v mod 3. For each coordinate i, the number of pairs (a_i, c_i)
    # with a_i + c_i = 2*v_i mod 3:
    #   if v_i = 0: pairs (0,0), (1,2), (2,1) -> 3 pairs
    #   if v_i = 1: pairs (0,2), (1,1), (2,0) -> 3 pairs  
    #   if v_i = 2: pairs (0,1), (1,0), (2,2) -> 3 pairs
    # Actually all coordinates have 3 pairs, so this doesn't differentiate.
    # Instead, use the sum of coordinates: vectors with sum closer to n (middle)
    # are more likely to be middles of 3-APs in aggregate.
    coord_sum = sum(vector)
    # Deviation from center (n/2 * 1.0 = n/2 if uniform... actually center is n*1 = n for mean value 1)
    # Mean coordinate value is 1, so center sum is n
    center_dev = abs(coord_sum - n)
    middle_risk = -center_dev  # lower deviation from center = higher risk = lower score
    
    # FNV-1a hash for tie-breaking
    h = 14695981039346656037
    for x in vector:
        h ^= x
        h = (h * 1099511628211) % (2**64)
    hash_score = float(h) * 1e-10  # scale down to be tie-breaker only
    
    return balance_score + middle_risk * 0.5 + hash_score
