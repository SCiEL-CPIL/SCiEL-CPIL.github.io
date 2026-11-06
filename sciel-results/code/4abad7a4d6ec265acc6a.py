# EVOLVE-BLOCK-START
import numpy as np

def construct():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    low = 1001
    high = 1000024
    size = high - low + 1
    
    nums = np.arange(low, high + 1, dtype=np.int64)
    smooth_parts = np.ones(size, dtype=np.int64)
    
    for p in primes:
        val = np.zeros(size, dtype=np.int32)
        power = p
        while power <= high:
            mask = (nums % power == 0)
            val[mask] += 1
            power *= p
        
        # Multiply smooth_parts by p^val
        # To avoid creating large temporary arrays, we can do:
        # smooth_parts *= (p ** val)
        # But p ** val creates a new array.
        # We can use np.power(p, val, out=smooth_parts) ? No, that overwrites.
        # We need smooth_parts = smooth_parts * (p ** val)
        
        # Optimization: 
        # smooth_parts *= p ** val is fine.
        
    # Direct frequency array
    # Max smooth part is high
    max_val = high
    freq = np.zeros(max_val + 1, dtype=np.int32)
    
    # Initialize window for n=1000 (indices 0 to 23)
    distinct_count = 0
    for i in range(24):
        sp = smooth_parts[i]
        if freq[sp] == 0:
            distinct_count += 1
        freq[sp] += 1
        
    best_count = distinct_count
    best_n = 1000
    
    # Slide window
    for n in range(1001, 1000001):
        out_idx = n - 1001
        sp_out = smooth_parts[out_idx]
        freq[sp_out] -= 1
        if freq[sp_out] == 0:
            distinct_count -= 1
            
        in_idx = n - 977
        sp_in = smooth_parts[in_idx]
        if freq[sp_in] == 0:
            distinct_count += 1
        freq[sp_in] += 1
            
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                break
                
    return best_n
# EVOLVE-BLOCK-END
