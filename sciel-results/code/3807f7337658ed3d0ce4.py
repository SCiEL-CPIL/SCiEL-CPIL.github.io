import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Range for n is [1000, 1000000]
    # We need n+1 to n+24, so max value is 1000024
    max_val = 1000024
    min_n = 1000
    max_n = 1000000
    
    # Initialize smooth parts array
    # smooth[i] will store the smooth part of i
    smooth = [1] * (max_val + 1)
    
    # Compute smooth parts using a sieve-like method
    for p in primes:
        for multiple in range(p, max_val + 1, p):
            val = multiple
            while val % p == 0:
                val //= p
                smooth[multiple] *= p
                
    # Sliding window to find minimum distinct smooth parts
    window_size = 24
    freq = {}
    distinct_count = 0
    min_distinct = float('inf')
    best_n = min_n
    
    # Initialize first window: n = min_n, elements are min_n+1 to min_n+24
    for i in range(min_n + 1, min_n + 1 + window_size):
        s = smooth[i]
        if s not in freq:
            freq[s] = 0
        freq[s] += 1
        if freq[s] == 1:
            distinct_count += 1
            
    if distinct_count < min_distinct:
        min_distinct = distinct_count
        best_n = min_n
        
    # Slide window
    for n in range(min_n + 1, max_n + 1):
        # Check time
        if time.time() > deadline:
            break
            
        # Remove element n (which was n-1 + 1, i.e., the leftmost element of previous window)
        # Previous window was [n, n+23] for n-1? No.
        # Window for n-1 was [(n-1)+1, (n-1)+24] = [n, n+23].
        # Window for n is [n+1, n+24].
        # So we remove smooth[n] and add smooth[n+24].
        
        old_val = smooth[n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            distinct_count -= 1
            
        new_val = smooth[n + window_size]
        if new_val not in freq:
            freq[new_val] = 0
        freq[new_val] += 1
        if freq[new_val] == 1:
            distinct_count += 1
            
        if distinct_count < min_distinct:
            min_distinct = distinct_count
            best_n = n
            
    return best_n
