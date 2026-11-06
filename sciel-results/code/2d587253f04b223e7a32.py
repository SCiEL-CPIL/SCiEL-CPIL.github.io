from collections import Counter

def construct():
    MAX_N = 1000000
    MAX_M = MAX_N + 24
    primes_below_24 = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    smooth = [1] * (MAX_M + 1)
    for p in primes_below_24:
        for multiple in range(p, MAX_M + 1, p):
            count = 0
            temp = multiple
            while temp % p == 0:
                count += 1
                temp //= p
            if count > 0:
                smooth[multiple] *= p ** count
    
    best_n = 1000
    best_count = 25
    
    # Use a list of counts indexed by the smooth value
    # Max smooth value is <= MAX_M
    counts = [0] * (MAX_M + 1)
    distinct_count = 0
    
    # Initialize window for n=1000: indices 1001 to 1024
    for i in range(1001, 1025):
        val = smooth[i]
        counts[val] += 1
        if counts[val] == 1:
            distinct_count += 1
            
    best_count = distinct_count
    best_n = 1000
    
    for n in range(1001, MAX_N + 1):
        # Remove smooth[n] (index n)
        val_remove = smooth[n]
        counts[val_remove] -= 1
        if counts[val_remove] == 0:
            distinct_count -= 1
            
        # Add smooth[n+24] (index n+24)
        val_add = smooth[n + 24]
        counts[val_add] += 1
        if counts[val_add] == 1:
            distinct_count += 1
            
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                break
                
    return best_n
