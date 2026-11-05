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
    
    # Precompute extracted[m] = product of p^k for p < 24 where p^k || m
    # Use a more efficient sieve approach
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in small_primes:
        # For each multiple of p, compute the p-part
        # m = p * k, p-part is p^v where v is the exponent of p in m
        multiples = np.arange(p, MAX_VAL + 1, p)
        # Compute p-part: for m = p*k, repeatedly divide k by p
        k = multiples // p  # k = m / p
        p_part = np.ones(len(multiples), dtype=np.int64)
        
        # Compute the exponent of p in each multiple
        # m = p * k, so v_p(m) = 1 + v_p(k)
        # We can compute this iteratively
        current = k.copy()
        exp_count = np.ones(len(multiples), dtype=np.int64)
        while np.any(current % p == 0):
            mask = current % p == 0
            current[mask] //= p
            exp_count[mask] += 1
        
        # p_part = p ^ exp_count
        # Use numpy power
        p_part = p ** exp_count
        extracted[multiples] *= p_part
    
    # Convert to list for faster access in the search loop
    extracted_list = extracted.tolist()
    
    best_n = N_MIN
    best_count = WINDOW + 1
    
    # Sliding window with dictionary
    counts = {}
    distinct_count = 0
    
    # Initialize window for n = N_MIN: indices N_MIN+1 to N_MIN+24
    for i in range(1, WINDOW + 1):
        val = extracted_list[N_MIN + i]
        if val in counts:
            counts[val] += 1
        else:
            counts[val] = 1
            distinct_count += 1
            
    if distinct_count < best_count:
        best_count = distinct_count
        best_n = N_MIN
        
    # Slide window
    for n in range(N_MIN + 1, N_MAX + 1):
        if time.time() - start_time > DEADLINE:
            break
            
        # Remove index n (leaving window for n-1)
        out_val = extracted_list[n]
        counts[out_val] -= 1
        if counts[out_val] == 0:
            distinct_count -= 1
            
        # Add index n+24 (entering window for n)
        in_val = extracted_list[n + WINDOW]
        if in_val in counts:
            counts[in_val] += 1
        else:
            counts[in_val] = 1
            distinct_count += 1
            
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                break
                
    return best_n
