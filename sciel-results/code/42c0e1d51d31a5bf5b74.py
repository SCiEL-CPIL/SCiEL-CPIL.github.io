import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    MAX_N = 1000000
    # We need smooth parts for n+1 to n+24, so up to MAX_N + 24
    # n is in [1000, MAX_N], so max index is MAX_N + 24
    # Allocate MAX_N + 25 to be safe (1-indexed logic or just size)
    max_val = MAX_N + 24
    smooth = np.ones(max_val + 1, dtype=np.int64)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Compute 23-smooth parts
    # smooth[x] will hold the largest divisor of x composed of primes in `primes`
    for p in primes:
        power = p
        while power <= max_val:
            # Multiply by p for all indices that are multiples of power
            # This accumulates the p-part
            smooth[power::power] *= p
            power *= p
            
    best_n = 1000
    best_count = 25  # Initial high value
    
    # Search
    # We iterate n from 1000 to MAX_N
    # The window is smooth[n+1] to smooth[n+24]
    # Indices in smooth array: n+1 to n+24 inclusive
    
    # To speed up the Python loop, we can convert the numpy array to a list? 
    # No, numpy slicing is fast, but set(window) requires iteration.
    # Converting to list might be faster for set() construction?
    # Let's try keeping it as numpy and converting to list inside the loop, 
    # or just using the numpy array directly in set().
    # set(np_array) works but iterates in Python.
    
    # Optimization: If best_count is 1, break.
    
    for n in range(1000, MAX_N + 1):
        if time.time() > deadline:
            break
        
        # Extract the 24 values
        # Slicing numpy array creates a view, set() iterates over it
        window = smooth[n+1:n+25]
        
        # Count distinct values
        # Using a set is the most direct way
        count = len(set(window))
        
        if count < best_count:
            best_count = count
            best_n = n
            if best_count == 1:
                # Cannot do better than 1 distinct factor
                break
                
    return best_n
