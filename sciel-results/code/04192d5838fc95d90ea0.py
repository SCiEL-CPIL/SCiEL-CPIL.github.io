import sys
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # Safety margin: The limit is 90s. We aim to finish in < 5s.
    deadline = start_time + 85.0
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24
    size = max_val + 1
    
    # Initialize small_part array using NumPy for speed
    # small_part[i] will hold the factor of i composed of primes < 24
    small_part = np.ones(size + 1, dtype=np.int64)
    
    # Sieve-like computation
    # For each prime p, for each multiple j of p, extract all powers of p from j
    # and multiply them into small_part[j]
    for p in primes:
        for j in range(p, size + 1, p):
            val = j
            while val % p == 0:
                val //= p
                small_part[j] *= p
    
    # Convert to Python list for faster access in the main loop
    # NumPy indexing is slower than list indexing for scalar access in Python loops
    sp = small_part.tolist()
    
    best_n = 1000
    best_count = 25  # Max possible distinct factors is 24
    
    # We will slide a window of size 24 over the range [1001, 1000000+24]
    # The window for n is sp[n+1 : n+25]
    
    # Initialize counts for the first window n=1000
    # Window elements: sp[1001] to sp[1024]
    counts = {}
    distinct_count = 0
    
    # Helper to add an element to the window counts
    def add_elem(val):
        nonlocal distinct_count
        if val not in counts:
            counts[val] = 1
            distinct_count += 1
        else:
            counts[val] += 1
            
    # Helper to remove an element from the window counts
    def remove_elem(val):
        nonlocal distinct_count
        counts[val] -= 1
        if counts[val] == 0:
            del counts[val]
            distinct_count -= 1
        # Note: counts[val] should never be negative if logic is correct
        
    # Initialize the first window
    for i in range(1001, 1001 + 24):
        add_elem(sp[i])
        
    if distinct_count < best_count:
        best_count = distinct_count
        best_n = 1000
        
    # Slide the window
    # n goes from 1001 to 1000000
    # For n, the window is sp[n+1 ... n+24]
    # For n-1, the window was sp[n ... n+23]
    # So we remove sp[n] and add sp[n+24]
    
    for n in range(1001, max_n + 1):
        # Remove the element leaving the window: sp[n]
        # The previous window (n-1) ended at (n-1)+24 = n+23
        # It started at (n-1)+1 = n
        # So sp[n] is the first element of the previous window, which is leaving now?
        # Let's re-verify indices.
        # Window for n: indices n+1, ..., n+24.
        # Window for n-1: indices n, ..., n+23.
        # To go from n-1 to n:
        # Remove index n.
        # Add index n+24.
        
        remove_elem(sp[n])
        add_elem(sp[n + 24])
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            
        # Optional: Early break if we reach theoretical minimum?
        # Theoretical minimum is 1 (if all 24 numbers have the same small part).
        # This is extremely unlikely. 2 is also unlikely.
        # We can break if best_count == 1.
        if best_count <= 1:
            break
        
        # Time check
        # The loop is very fast, but just in case
        if (n - 1001) % 100000 == 0:
            if time.time() > deadline:
                break

    return best_n
# EVOLVE-BLOCK-END
