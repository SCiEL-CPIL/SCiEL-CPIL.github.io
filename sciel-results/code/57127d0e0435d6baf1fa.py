import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Compute small_factor[x] = product of p^v_p(x) for all primes p < 24
    # Using a more efficient sieve approach with numpy
    
    small_factor = np.ones(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        # For each multiple of p, compute p^v_p(m)
        # Use vectorized operations where possible
        multiples = np.arange(p, max_val + 1, p)
        # For each multiple, count how many times p divides it
        # We need to compute p^e where e = v_p(m)
        # Start with all multiples, divide by p repeatedly
        
        # Efficient approach: for each power p^k, mark multiples of p^k
        # and multiply by p for each such marking
        power = p
        while power <= max_val:
            # All multiples of power get an additional factor of p
            indices = np.arange(power, max_val + 1, power)
            small_factor[indices] *= p
            power *= p
    
    # Convert to list for faster sliding window operations
    small_factor_list = small_factor.tolist()
    
    # Now sliding window scan
    freq = {}
    active_count = 0
    
    # Initialize for n=1000: window is [1001, 1024]
    for i in range(1, 25):
        val = small_factor_list[1000 + i]
        if val not in freq:
            freq[val] = 0
            active_count += 1
        freq[val] += 1
        
    best_score = active_count
    best_n = 1000
    
    # Slide
    for n in range(1001, max_n + 1):
        if time.time() > deadline:
            break
            
        old_val = small_factor_list[n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            active_count -= 1
            del freq[old_val]
            
        new_val = small_factor_list[n + 24]
        if new_val not in freq:
            freq[new_val] = 0
            active_count += 1
        freq[new_val] += 1
        
        if active_count < best_score:
            best_score = active_count
            best_n = n
            if best_score == 1:
                break
                
    return best_n
