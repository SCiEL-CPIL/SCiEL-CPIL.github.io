import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    min_n = 1000
    window_size = 24
    
    # Max value we need to factor is max_n + 24
    max_val = max_n + window_size
    
    # Precompute small factors for all numbers in range
    # small_part[x] is the product of prime powers p^k for p < 24 dividing x
    small_part = [1] * (max_val + 1)
    
    # Sieve-like computation
    # For each prime p, iterate through its multiples
    # This is more efficient than looping x and then p
    # But careful: we need to accumulate powers.
    
    # Method:
    # For each x in 1..max_val:
    #   temp = x
    #   factor = 1
    #   for p in primes:
    #      while temp % p == 0:
    #         factor *= p
    #         temp //= p
    #   small_part[x] = factor
    
    # Let's try the direct loop first, it's simple and likely fast enough for 10^6.
    for x in range(1, max_val + 1):
        temp = x
        factor = 1
        for p in primes:
            while temp % p == 0:
                factor *= p
                temp //= p
                if temp == 1:
                    break
        small_part[x] = factor

    # Sliding window
    # Initialize counts for n = min_n
    # Window is [min_n+1, min_n+24]
    counts = {}
    distinct_count = 0
    
    for i in range(1, window_size + 1):
        val = small_part[min_n + i]
        if val in counts:
            counts[val] += 1
        else:
            counts[val] = 1
            distinct_count += 1
            
    best_n = min_n
    best_score = distinct_count
    
    # Iterate n from min_n to max_n
    for n in range(min_n, max_n + 1):
        if distinct_count < best_score:
            best_score = distinct_count
            best_n = n
            
        # Slide to n+1
        # Remove n+1, add n+25
        if n < max_n: # Don't slide if we are at the last valid n
            out_val = small_part[n + 1]
            in_val = small_part[n + window_size + 1]
            
            # Remove out_val
            counts[out_val] -= 1
            if counts[out_val] == 0:
                distinct_count -= 1
                # Optional: del counts[out_val] to keep dict small, but not strictly necessary if we check == 0
                # However, keeping it might be slightly faster than del+add later if it reappears?
                # Let's just let it stay 0 or delete it. Deleting is cleaner.
                del counts[out_val]
                
            # Add in_val
            if in_val in counts:
                counts[in_val] += 1
            else:
                counts[in_val] = 1
                distinct_count += 1

        # Time check (should be fast, but just in case)
        if time.time() > deadline:
            break

    return best_n
