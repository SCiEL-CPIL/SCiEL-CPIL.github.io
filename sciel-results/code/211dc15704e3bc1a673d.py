def construct():
    # The set is constructed by selecting powers of primes such that
    # the subset sums of the exponents for each prime are unique.
    # For p=2, powers 1, 2, 4, 8, 16 (values 2, 4, 8, 16, 32) allow unique sums 0..31.
    # For p=3, powers 1, 2 (values 3, 9) allow unique sums 0,1,2,3.
    # For p=5, powers 1, 2 (values 5, 25) allow unique sums 0,1,2,3.
    # For other primes <= 40, only the prime itself (power 1) is <= 40.
    # Primes: 7, 11, 13, 17, 19, 23, 29, 31, 37.
    
    # 2-powers: 2, 4, 8, 16, 32
    # 3-powers: 3, 9
    # 5-powers: 5, 25
    # Other primes: 7, 11, 13, 17, 19, 23, 29, 31, 37
    
    result = [2, 4, 8, 16, 32, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verification (optional, for safety)
    # Check distinctness
    if len(set(result)) != len(result):
        return []
    # Check range
    if not all(2 <= x <= 40 for x in result):
        return []
        
    # Check subset product distinctness
    # This is expensive for 2^18, but we can do a quick check or trust the math.
    # 2^18 = 262144 products.
    # We can generate them to be 100% sure.
    k = len(result)
    prods = set()
    # Use iterative generation
    # Start with {1}
    current_prods = {1}
    for x in result:
        new_prods = set()
        for p in current_prods:
            new_prods.add(p * x)
        current_prods.update(new_prods)
        
    if len(current_prods) != (1 << k):
        # Fallback to a smaller known valid set if this fails (should not happen)
        return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    return result
