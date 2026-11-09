import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    n_min = 1000
    n_max = 1000000
    
    val_min = n_min + 1
    val_max = n_max + 24
    
    size = n_max - n_min + 1  # number of n values
    total_vals = val_max - val_min + 1  # number of values to compute G for
    
    # Precompute G using vectorized operations
    indices = np.arange(val_min, val_max + 1, dtype=np.int64)
    G = np.ones(total_vals, dtype=np.int64)
    
    for p in primes:
        p_power = p
        while p_power <= val_max:
            mask = (indices % p_power == 0)
            G[mask] *= p
            p_power *= p
            if time.time() > deadline:
                return 1000
    
    G_list = G.tolist()
    
    freq = defaultdict(int)
    
    # Initialize first window: positions 0..23 in G correspond to n_min+1..n_min+24
    active_count = 0
    for i in range(24):
        v = G_list[i]
        if freq[v] == 0:
            active_count += 1
        freq[v] += 1
        
    min_distinct = active_count
    best_n = n_min
    
    # Slide window
    for i in range(1, size):
        val_out = G_list[i - 1]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            active_count -= 1
            
        val_in = G_list[i + 23]
        if freq[val_in] == 0:
            active_count += 1
        freq[val_in] += 1
        
        if active_count < min_distinct:
            min_distinct = active_count
            best_n = n_min + i
            if min_distinct <= 2:  # Can't do much better than 2
                break
        
        if i % 50000 == 0 and time.time() > deadline:
            break
    
    return int(best_n)
