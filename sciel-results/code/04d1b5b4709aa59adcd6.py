import time

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_n = 1000000
    max_val = max_n + 24
    
    # Precompute bitmask of small prime factors for each number up to max_val
    # mask[m] will have bit i set if primes[i] divides m
    mask = [0] * (max_val + 1)
    
    for i, p in enumerate(primes):
        bit = 1 << i
        for m in range(p, max_val + 1, p):
            mask[m] |= bit
    
    best_n = 1000
    min_count = 10  # Max possible is 9, so 10 is a safe initial value
    
    # Iterate through all valid n
    for n in range(1000, max_n + 1):
        union_mask = 0
        for i in range(1, 25):
            union_mask |= mask[n + i]
        
        # Count set bits
        count = bin(union_mask).count('1')
        
        if count < min_count:
            min_count = count
            best_n = n
            # If we find 1 or 2, it's likely optimal, but let's check all to be sure 
            # or break if we hit a theoretical lower bound? 
            # Theoretically, can we have 1? No. 
            # Can we have 2? Maybe. 
            # But 3 or 4 is more likely. 
            # We should probably run the full loop to be certain, 
            # but if time is a concern, we can optimize. 
            # 10^6 * 24 operations is 24 * 10^6, which is very fast in Python 
            # if we use bit operations. 
            # Actually, Python loops are slow. 24 million iterations might take a few seconds. 
            # Let's see. 
            # We can break early if min_count becomes 1? Unlikely. 
            # We can break if min_count becomes 2? Possible but rare. 
            # Let's just run it all. It should be under 1 second. 
            pass
            
    return best_n
