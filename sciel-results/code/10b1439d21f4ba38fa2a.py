import time

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)
    
    # Maximum value needed is 1000000 + 24 = 1000024
    max_val = 1000024
    masks = [0] * (max_val + 1)
    
    # Precompute masks for all numbers up to max_val
    # masks[m] is a bitmask where bit j is 1 if primes[j] divides m
    for j, p in enumerate(primes):
        bit = 1 << j
        for m in range(p, max_val + 1, p):
            masks[m] |= bit
    
    # Sliding window logic
    # We want to minimize the number of distinct prime factors in the union of small parts of n+1 ... n+24.
    # The union of prime factors is represented by the bitwise OR of the masks.
    # We maintain the frequency of each bit in the current window to know when a bit disappears from the OR.
    
    window_size = 24
    freq = [0] * num_primes
    current_or_mask = 0
    
    # Initialize the first window: n=1000, numbers are 1001 to 1024
    start_n = 1000
    initial_end = start_n + window_size # exclusive, so numbers are start_n+1 to start_n+24
    
    for m in range(start_n + 1, initial_end + 1):
        mask = masks[m]
        for j in range(num_primes):
            if mask & (1 << j):
                freq[j] += 1
                current_or_mask |= (1 << j)
    
    min_count = bin(current_or_mask).count('1')
    best_n = start_n
    
    # Slide the window
    # Next n is 1001, numbers are 1002 to 1025
    # We remove 1001 and add 1025
    for n in range(start_n + 1, 1000001 - window_size + 1):
        # Remove the outgoing number: n
        out_m = n
        out_mask = masks[out_m]
        for j in range(num_primes):
            if out_mask & (1 << j):
                freq[j] -= 1
                if freq[j] == 0:
                    current_or_mask &= ~(1 << j)
        
        # Add the incoming number: n + window_size
        in_m = n + window_size
        in_mask = masks[in_m]
        for j in range(num_primes):
            if in_mask & (1 << j):
                freq[j] += 1
                current_or_mask |= (1 << j)
        
        # Calculate current count
        count = bin(current_or_mask).count('1')
        
        if count < min_count:
            min_count = count
            best_n = n
            
    return best_n
