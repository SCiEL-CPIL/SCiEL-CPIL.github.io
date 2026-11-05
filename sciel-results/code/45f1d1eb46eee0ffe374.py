import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0

    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW

    # --- Step 1: Precompute the smooth part (product of prime powers for primes < 24) ---
    # Using a sieve-like approach with NumPy for speed.
    # extracted[m] = product of p^k for all primes p < 24 such that p^k divides m.
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in small_primes:
        multiples = np.arange(p, MAX_VAL + 1, p)
        p_parts = np.ones(len(multiples), dtype=np.int64)
        temp = multiples.copy()
        
        # Calculate p-part for all multiples efficiently
        # We divide by p repeatedly until no longer divisible
        while True:
            mask = temp % p == 0
            if not np.any(mask):
                break
            temp[mask] //= p
            p_parts[mask] *= p
        extracted[multiples] *= p_parts

    # --- Step 2: Vectorized sliding window distinct count ---
    # We want to find n in [N_MIN, N_MAX] such that the number of distinct values
    # in the set {extracted[n+1], ..., extracted[n+24]} is minimized.
    
    # Convert to int32 if possible to save memory? 
    # Max smooth part for m <= 1,000,024.
    # Max power of 2 is 2^19 = 524288.
    # Max power of 3 is 3^12 = 531441.
    # Product could be larger, but since it's a divisor of m, it is <= m <= 1,000,024.
    # So int32 is sufficient and safe.
    # However, np.unique returns indices, so we need to handle mapping carefully.
    
    # Let's stick to int64 or let numpy handle it. 
    # To make the vectorized distinct count fast, we can use a trick:
    # For each window, we want the count of unique elements.
    # Doing this for 1 million windows of size 24 using np.unique in a loop is too slow (1M * sort(24)).
    
    # Alternative:
    # The current Python loop was 1.4286.
    # Let's try to optimize the Python loop further by using local variables and avoiding defaultdict overhead.
    # Or, use a different vectorization strategy.
    
    # Strategy: 
    # We can process the array in chunks if we could, but the window is small.
    # Let's try to speed up the Python loop significantly.
    # The main cost is the sliding window update.
    # Using a list for frequencies is faster than defaultdict.
    # But the values can be up to 1,000,000. We can use a list of size MAX_VAL+1.
    
    extracted_list = extracted.tolist()
    
    best_n = N_MIN
    best_count = WINDOW + 1
    
    # Frequency array for values. Since values are <= MAX_VAL, we can use a list.
    # But allocating a list of 1M ints is fine.
    freq = [0] * (MAX_VAL + 1)
    distinct = 0
    
    start_idx = N_MIN + 1
    end_idx = start_idx + WINDOW - 1
    
    # Initialize first window
    for i in range(start_idx, end_idx + 1):
        val = extracted_list[i]
        if freq[val] == 0:
            distinct += 1
        freq[val] += 1
        
    if distinct < best_count:
        best_count = distinct
        best_n = N_MIN
        
    # Slide the window
    # n goes from N_MIN + 1 to N_MAX
    # Window for n is [n+1, n+24]
    # Window for n-1 is [n, n+23]
    # Remove n, Add n+24
    
    # To speed up, we can try to use numpy for the comparison?
    # No, the stateful nature of distinct count makes it hard to vectorize directly without complex tricks.
    
    # Let's stick to the optimized Python loop.
    # Optimization: Local variables for speed.
    el = extracted_list
    fr = freq
    wc = WINDOW
    nm = N_MAX
    t0 = start_time
    dl = DEADLINE
    
    for n in range(N_MIN + 1, nm + 1):
        # Time check every few iterations to avoid overhead?
        # time.time() is relatively cheap, but let's check every 1000 iterations or just use the simple check.
        if (n & 1023) == 0: # Check every 1024 iterations
            if time.time() - t0 > dl:
                break
            
        val_out = el[n]
        fr[val_out] -= 1
        if fr[val_out] == 0:
            distinct -= 1
            
        val_in = el[n + wc]
        if fr[val_in] == 0:
            distinct += 1
        fr[val_in] += 1
        
        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count == 1:
                break

    return best_n
