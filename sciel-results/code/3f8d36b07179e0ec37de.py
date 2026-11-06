"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.time()
    deadline = start_time + 45.0

    lower = 1000
    upper = 1000000
    window_size = 24
    max_val = upper + window_size
    
    nvals = max_val - lower + 1
    
    # Compute smooth parts
    sp_part = [1] * nvals
    
    for p in _SMALL_PRIMES:
        first_mult = ((lower + p - 1) // p) * p
        for m in range(first_mult, max_val + 1, p):
            idx = m - lower
            temp = m
            while temp % p == 0:
                sp_part[idx] *= p
                temp //= p
                
    if time.time() > deadline:
        return lower

    # Map smooth parts to IDs for faster frequency counting
    # The number of distinct smooth parts is relatively small
    id_map = {}
    next_id = 0
    sp_ids = [0] * nvals
    for i in range(nvals):
        val = sp_part[i]
        if val not in id_map:
            id_map[val] = next_id
            next_id += 1
        sp_ids[i] = id_map[val]
        
    num_ids = next_id
    
    # Sliding window
    best_n = lower
    best_count = float('inf')
    
    freq = [0] * num_ids
    current_distinct = 0
    
    # Initialize window for n=lower (indices 1 to 24)
    for i in range(1, window_size + 1):
        val = sp_ids[i]
        if freq[val] == 0:
            current_distinct += 1
        freq[val] += 1
        
    if current_distinct < best_count:
        best_count = current_distinct
        best_n = lower
        
    # Slide window
    for n in range(lower + 1, upper + 1):
        if time.time() > deadline:
            break
            
        out_idx = n - lower
        in_idx = n + 24 - lower
        
        val_out = sp_ids[out_idx]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            current_distinct -= 1
            
        val_in = sp_ids[in_idx]
        if freq[val_in] == 0:
            current_distinct += 1
        freq[val_in] += 1
        
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
            # If we reach a very low count, we might want to stop?
            # But we want the global min.
            # However, if best_count is 1, 2, etc., it's very good.
            # We must finish to ensure no lower count exists.
            # But 1 is impossible. 2 is very unlikely.
            # We can stop if best_count == 2? No, 1 is impossible, so 2 is the min possible?
            # Wait, 1 is impossible. So 2 is the theoretical min.
            # If we find 2, we can stop.
            if best_count == 2:
                # Can we get 1? No. So 2 is global min.
                # But wait, is 2 possible?
                # If yes, stop.
                # If not, we'll finish the scan.
                # To be safe, don't stop unless we are sure 1 is impossible.
                # We proved 1 is impossible.
                # So if we find 2, it's the best possible.
                # Let's stop.
                break

    return best_n
# EVOLVE-BLOCK-END
