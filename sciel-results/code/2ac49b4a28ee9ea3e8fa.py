def construct():
    import time
    start_time = time.time()
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Map each prime to a bit position
    # 2 -> 0, 3 -> 1, 5 -> 2, etc.
    prime_to_bit = {p: 1 << i for i, p in enumerate(primes)}
    
    max_n = 1000000
    max_val = max_n + 24
    
    # Precompute small prime factor masks for all numbers up to max_val
    # factors[x] will be a bitmask of primes in 'primes' that divide x
    factors = [0] * (max_val + 1)
    
    for p in primes:
        bit = prime_to_bit[p]
        for multiple in range(p, max_val + 1, p):
            factors[multiple] |= bit
    
    # Lookup table for popcount of 8-bit integers
    popcount_8 = [0] * 256
    for i in range(256):
        popcount_8[i] = bin(i).count('1')
        
    def popcount(x):
        # x is at most 9 bits (9 primes)
        return popcount_8[x & 0xFF] + popcount_8[(x >> 8) & 0xFF]
        
    min_count = 100
    best_n = 1000
    
    # Iterate through all valid n
    for n in range(1000, max_n + 1):
        mask = 0
        # Unroll or use a simple loop
        # n+1 to n+24
        for i in range(1, 25):
            mask |= factors[n + i]
            
        count = popcount(mask)
        
        if count < min_count:
            min_count = count
            best_n = n
            
        # Time check
        if time.time() - start_time > 80:
            break
            
    return best_n
