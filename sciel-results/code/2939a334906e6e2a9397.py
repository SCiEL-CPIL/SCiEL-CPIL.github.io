import time

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_val = 1000000 + 24
    
    # Precompute small prime parts using sieve-style multiplication
    # small_parts[m] = product of p^v_p(m) for p in primes
    small_parts = [1] * (max_val + 1)
    
    for p in primes:
        pk = p
        while pk <= max_val:
            for m in range(pk, max_val + 1, pk):
                small_parts[m] *= p
            pk *= p
    
    best_n = 1000
    best_score = 25  # Max possible distinct values is 24
    
    # Sliding window approach
    # Window is [n+1, n+24]
    # We start with n=1000, window is [1001, 1024]
    
    # Initialize counts for the first window
    counts = {}
    for i in range(1, 25):
        val = small_parts[1000 + i]
        counts[val] = counts.get(val, 0) + 1
        
    distinct_count = len(counts)
    
    if distinct_count < best_score:
        best_score = distinct_count
        best_n = 1000
        
    start_time = time.monotonic()
    deadline = start_time + 75  # Safety margin before 90s kill
    
    # Iterate n from 1001 to 1000000
    # Window for n is [n+1, n+24]
    # Previous window for n-1 was [n, n+23]
    # So we remove n, add n+24
    
    for n in range(1001, 1000001):
        if time.monotonic() > deadline:
            break
            
        # Remove element at n (which was n-1 + 1)
        val_out = small_parts[n]
        c = counts[val_out] - 1
        counts[val_out] = c
        if c == 0:
            distinct_count -= 1
            
        # Add element at n+24
        val_in = small_parts[n + 24]
        c = counts.get(val_in, 0)
        if c == 0:
            distinct_count += 1
        counts[val_in] = c + 1
        
        if distinct_count < best_score:
            best_score = distinct_count
            best_n = n
            if best_score == 1:
                break
                
    return best_n
