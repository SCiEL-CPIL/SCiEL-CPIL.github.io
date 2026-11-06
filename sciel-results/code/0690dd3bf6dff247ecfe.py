"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    from collections import defaultdict
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    min_x = 1001
    max_x = 1000024
    size = max_x - min_x + 1
    
    sp_arr = np.ones(size, dtype=np.int64)
    
    for idx in range(size):
        x = min_x + idx
        val = x
        for p in small_primes:
            while val % p == 0:
                val //= p
        sp_arr[idx] = x // val
    
    freq = defaultdict(int)
    for i in range(24):
        freq[sp_arr[i]] += 1
    
    distinct_count = len(freq)
    best_count = distinct_count
    best_n = 1000
    
    for n in range(1001, 1000001):
        old_idx = n - 1001
        new_idx = n - 977
        
        old_val = sp_arr[old_idx]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
        
        new_val = sp_arr[new_idx]
        freq[new_val] += 1
        
        distinct_count = len(freq)
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
