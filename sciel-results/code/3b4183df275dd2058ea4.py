import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin (90s limit)
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    min_n = 1000
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Precompute sf[m] = product of p^k for primes p<24, p^k || m
    # sf[m] is the "small prime part"
    sf = np.ones(max_val + 1, dtype=np.int64)
    
    for p in primes:
        # For each power of p, multiply sf[m] by p for all m divisible by p^k
        pk = p
        while pk <= max_val:
            # For all multiples of pk, multiply sf by p
            # This handles the full power: sf[m] gets multiplied by p for each p^k dividing m
            multiples = np.arange(pk, max_val + 1, pk)
            sf[multiples] *= p
            pk *= p
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break
    
    # Now for each n in [1000, 1000000], count distinct sf values in sf[n+1..n+24]
    # We want to minimize this count
    
    best_n = min_n
    best_count = 25  # max possible is 24 distinct values
    
    # Use a sliding window approach with a counter
    # For efficiency, use numpy but distinct counting is tricky
    
    # Let's iterate and use a set for the window
    # But that's slow in Python. Let's think of a better way.
    
    # Alternative: for each n, extract sf[n+1:n+25] and count unique
    # With numpy, np.unique is fast but we do it 1M times... too slow.
    
    # Better: use a frequency array approach with sliding window
    # But sf values can be large, so we need to map them.
    
    # Actually, let's just do a direct scan with numpy for chunks
    # Or: iterate n, but use a dictionary for the window
    
    # Given time constraints, let's do a smarter approach:
    # Precompute sf as a numpy array, then for each n, get the 24 values and count unique.
    # 1M iterations with numpy unique on 24 elements might be ~10-20s
    
    best_count = 25
    best_n = min_n
    
    # Process in chunks to allow deadline checking
    chunk_size = 10000
    for start in range(min_n, max_n + 1, chunk_size):
        end = min(start + chunk_size, max_n + 1)
        if time.time() > deadline - 3:
            break
        
        for n in range(start, end):
            window = sf[n+1:n+25]
            # Count distinct values
            # For small arrays, np.unique is fine
            cnt = len(np.unique(window))
            if cnt < best_count:
                best_count = cnt
                best_n = n
                if best_count == 1:
                    return best_n
            if time.time() > deadline - 3:
                break
    
    return best_n
