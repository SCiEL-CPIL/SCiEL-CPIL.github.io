import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    n_min = 1000
    n_max = 1000000
    
    val_min = n_min + 1
    val_max = n_max + 24
    
    size = n_max - n_min + 1
    
    # Precompute G
    indices = np.arange(val_min, val_max + 1, dtype=np.int64)
    G = np.ones(len(indices), dtype=np.int64)
    
    for p in primes:
        p_power = p
        while p_power <= val_max:
            mask = (indices % p_power == 0)
            G[mask] *= p
            p_power *= p
            if time.time() > deadline:
                return 1000
                
    # Convert to list for faster access in Python loop? 
    # Actually, numpy array access is okay.
    # But let's try to use a list if it's faster.
    # G_list = G.tolist() # This takes time but might be faster in loop?
    # Let's stick to numpy array for now, but optimize the loop.
    
    freq = defaultdict(int)
    
    # Initialize first window
    active_count = 0
    for i in range(24):
        v = G[i]
        if freq[v] == 0:
            active_count += 1
        freq[v] += 1
        
    min_distinct = active_count
    best_n = n_min
    
    # Slide
    # We need to access G[i+23] and G[i-1]
    # G is 1-indexed relative to val_min? No, 0-indexed.
    # G[0] is val_min.
    
    for i in range(1, size):
        # Remove G[i-1]
        val_out = G[i-1]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            active_count -= 1
            
        # Add G[i+23]
        val_in = G[i+23]
        if freq[val_in] == 0:
            active_count += 1
        freq[val_in] += 1
        
        if active_count < min_distinct:
            min_distinct = active_count
            best_n = n_min + i
            
        if time.time() > deadline:
            break
            
    return int(best_n)
