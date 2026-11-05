import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW
    
    # Precompute extracted[m] = product of p^k for p < 24 where p^k || m
    # Using NumPy for speed
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in small_primes:
        multiples = np.arange(p, MAX_VAL + 1, p)
        p_parts = np.ones(len(multiples), dtype=np.int64)
        temp = multiples.copy()
        # Efficiently compute p-part for all multiples at once
        # We can do this by repeated division, but numpy vectorization is key
        # Since p is small, the max exponent is small (e.g., for p=2, max exponent is ~20 for 10^6)
        # But doing a while loop with masks is fine as it converges quickly.
        while np.any(temp % p == 0):
            mask = temp % p == 0
            temp[mask] //= p
            p_parts[mask] *= p
        extracted[multiples] *= p_parts
    
    # Convert to list for faster access in the loop
    extracted_list = extracted.tolist()
    
    best_n = N_MIN
    best_count = WINDOW + 1  # worst case is 24 distinct
    
    # Sliding window distinct count
    # We want to evaluate windows [n+1, n+24] for n in [N_MIN, N_MAX]
    # This is equivalent to windows starting at S = n+1, where S in [N_MIN+1, N_MAX+1]
    # Let's slide a window of size WINDOW over extracted_list.
    
    # Initialize frequency for the first window: indices [N_MIN+1, N_MIN+WINDOW]
    freq = defaultdict(int)
    distinct = 0
    
    start_idx = N_MIN + 1
    end_idx = start_idx + WINDOW - 1
    
    for i in range(start_idx, end_idx + 1):
        val = extracted_list[i]
        if freq[val] == 0:
            distinct += 1
        freq[val] += 1
        
    if distinct < best_count:
        best_count = distinct
        best_n = N_MIN
        
    # Slide the window
    # Next n is N_MIN + 1. Window is [N_MIN+2, N_MIN+25]
    # Remove index N_MIN+1, Add index N_MIN+25
    for n in range(N_MIN + 1, N_MAX + 1):
        if time.time() - start_time > DEADLINE:
            break
            
        # Remove the element leaving the window: n
        # The previous window was [n, n+23] corresponding to n-1?
        # Let's trace carefully.
        # Window for n is [n+1, n+24].
        # Window for n-1 is [n, n+23].
        # So to go from n-1 to n, we remove n and add n+24.
        
        val_out = extracted_list[n]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            distinct -= 1
            # del freq[val_out] # Not strictly necessary if we check == 0, but good for memory if range was huge. Here it's fine.
            
        val_in = extracted_list[n + WINDOW]
        if freq[val_in] == 0:
            distinct += 1
        freq[val_in] += 1
        
        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count == 1:
                break  # Can't do better than 1
    
    return best_n
