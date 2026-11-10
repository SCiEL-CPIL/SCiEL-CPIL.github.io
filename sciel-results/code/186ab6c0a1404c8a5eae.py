import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Sieve-based computation of small-prime factors
    # small_factor[x] = product of p^v_p(x) for all small primes p
    # Use a more efficient approach: for each prime p, compute p^k contributions
    
    small_factor = [1] * (max_val + 1)
    
    for p in small_primes:
        # For each power p^k, mark all multiples
        pk = p
        while pk <= max_val:
            for multiple in range(pk, max_val + 1, pk):
                small_factor[multiple] *= p
            pk *= p
    
    # Now sliding window scan with frequency map
    # Use a list-based approach for speed since values are bounded
    
    # First, let's understand the range of possible small_factor values
    # Max small_factor: product of all small primes to their max power in range
    # This can be large, so we'll use a dict
    
    freq = {}
    active_count = 0
    
    # Initialize for n=1000: window is [1001, 1024]
    for i in range(1, 25):
        val = small_factor[1000 + i]
        if val not in freq:
            freq[val] = 0
            active_count += 1
        freq[val] += 1
        
    best_score = active_count
    best_n = 1000
    
    if best_score == 1:
        return best_n
    
    # Slide through all windows
    # Optimize: check time periodically, not every iteration
    check_interval = 10000
    for n in range(1001, max_n + 1):
        if (n - 1001) % check_interval == 0:
            if time.time() > deadline:
                break
        
        old_val = small_factor[n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            active_count -= 1
            del freq[old_val]
            
        new_val = small_factor[n + 24]
        if new_val not in freq:
            freq[new_val] = 0
            active_count += 1
        freq[new_val] += 1
        
        if active_count < best_score:
            best_score = active_count
            best_n = n
            if best_score == 1:
                break
            # If we found a very good score, we can be more aggressive with time checks
            if best_score <= 3:
                check_interval = 1000
    
    return best_n
