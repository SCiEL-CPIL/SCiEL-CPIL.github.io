import time
import sys

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    N = 1000000
    MAX_VAL = N + 24
    
    # Precompute the small part for all numbers up to MAX_VAL
    # small_part[x] will be the factor of x composed of primes in 'primes'
    sp = [1] * (MAX_VAL + 1)
    
    # Efficient precomputation using a sieve-like approach:
    # For each prime p, for each power p^k <= MAX_VAL,
    # multiply sp[m] by p for all m that are multiples of p^k.
    # This correctly accumulates p^v_p(x) for each x.
    # Complexity: sum over primes p of (MAX_VAL/p + MAX_VAL/p^2 + ...) ≈ MAX_VAL * sum(1/(p-1))
    # which is much faster than the O(N * P * log) direct factorization.
    
    for p in primes:
        power = p
        while power <= MAX_VAL:
            for m in range(power, MAX_VAL + 1, power):
                sp[m] *= p
            power *= p
    
    # Initialize sliding window for n = 1000
    # Window is sp[1001] ... sp[1024]
    start_n = 1000
    window = sp[start_n + 1 : start_n + 25]
    
    freq = {}
    for val in window:
        freq[val] = freq.get(val, 0) + 1
        
    distinct_count = len(freq)
    best_n = start_n
    min_distinct = distinct_count
    
    start_time = time.time()
    deadline = 85.0
    
    # Slide the window
    # Current window is for n. Next n is n+1.
    # Remove sp[n+1], Add sp[n+25] relative to the starting index of the window?
    # Let's stick to the derivation:
    # Window(n) = {sp[n+1], ..., sp[n+24]}
    # Window(n+1) = {sp[n+2], ..., sp[n+25]}
    # Diff: Remove sp[n+1], Add sp[n+25]
    
    for n in range(start_n + 1, N + 1):
        if (n & 1023) == 0:
            if time.time() - start_time > deadline:
                break
                
        out_val = sp[n]      # sp[(n-1)+1]
        in_val = sp[n + 24]  # sp[(n-1)+25] = sp[n+24]
                             
        # Update freq for out_val
        count_out = freq[out_val]
        if count_out == 1:
            del freq[out_val]
            distinct_count -= 1
        else:
            freq[out_val] = count_out - 1
            
        # Update freq for in_val
        count_in = freq.get(in_val, 0)
        if count_in == 0:
            freq[in_val] = 1
            distinct_count += 1
        else:
            freq[in_val] = count_in + 1
            
        if distinct_count < min_distinct:
            min_distinct = distinct_count
            best_n = n
            
            if min_distinct == 1:
                break
                
    return best_n

construct()
