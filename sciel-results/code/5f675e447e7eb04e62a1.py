import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill

    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(small_primes)
    
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW

    # Precompute bitmask: bit k set if prime[k] divides m
    # Also precompute the actual extracted factor value for tie-breaking
    # But first, let's use bitmasks for fast distinct counting
    
    # For each m, compute which small primes divide it
    # bitmask[m] = OR of (1<<k) for each k where small_primes[k] | m
    bitmask = np.zeros(MAX_VAL + 1, dtype=np.int32)
    
    for k, p in enumerate(small_primes):
        bitmask[p::p] |= (1 << k)
    
    # Now slide a window of size WINDOW over bitmask[N_MIN+1 : N_MAX+1+WINDOW]
    # For each position n, the window is bitmask[n+1 : n+1+WINDOW]
    # Count distinct bitmasks in each window
    
    # Use a frequency array approach:
    # For window starting at index i (0-indexed in the bitmask array),
    # elements are bitmask[i], bitmask[i+1], ..., bitmask[i+WINDOW-1]
    # We want count of distinct values
    
    # The bitmask values are 9-bit numbers (0 to 511), so we can use a small frequency array
    
    # Process in chunks for memory efficiency
    CHUNK_SIZE = 100000
    
    best_n = N_MIN
    best_count = WINDOW + 1  # worst case: all distinct
    
    total_starts = N_MAX - N_MIN + 1  # number of possible n values
    
    for chunk_start in range(0, total_starts, CHUNK_SIZE):
        if time.time() - start_time > DEADLINE:
            break
        
        chunk_end = min(chunk_start + CHUNK_SIZE, total_starts)
        n_start = N_MIN + chunk_start
        n_end = N_MIN + chunk_end  # exclusive
        
        # We need bitmask values from index n_start+1 to n_end+WINDOW-1
        # That's bitmask[n_start+1 : n_end+WINDOW]
        arr_start = n_start + 1
        arr_end = n_end + WINDOW
        
        flat = bitmask[arr_start : arr_end]
        chunk_len = chunk_end - chunk_start
        
        # For each window position j (0 to chunk_len-1), the window covers
        # flat[j : j+WINDOW]
        
        # Use sliding window with frequency counting
        # Initialize frequency for first window
        freq = np.zeros(512, dtype=np.int32)
        unique_count = 0
        
        for val in flat[:WINDOW]:
            v = int(val)
            if freq[v] == 0:
                unique_count += 1
            freq[v] += 1
        
        local_min_val = unique_count
        local_min_idx = 0
        
        for j in range(1, chunk_len):
            # Remove flat[j-1]
            v_out = int(flat[j - 1])
            freq[v_out] -= 1
            if freq[v_out] == 0:
                unique_count -= 1
            
            # Add flat[j+WINDOW-1]
            v_in = int(flat[j + WINDOW - 1])
            if freq[v_in] == 0:
                unique_count += 1
            freq[v_in] += 1
            
            if unique_count < local_min_val:
                local_min_val = unique_count
                local_min_idx = j
        
        if local_min_val < best_count:
            best_count = local_min_val
            best_n = n_start + local_min_idx
            if best_count == 1:
                break
    
    return best_n
