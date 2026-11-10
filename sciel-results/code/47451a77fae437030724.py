import time
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 90s. We aim to finish well before.
    deadline = start_time + 85
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24  # We need indices up to 1000024
    
    # Precompute small_factor[x] = product of p^v_p(x) for p < 24
    # Initialize with 1s
    small_factor = np.ones(max_val + 1, dtype=np.int64)
    
    # Sieve-like accumulation
    for p in small_primes:
        power = p
        while power <= max_val:
            # Indices: power, 2*power, 3*power, ...
            # For each index i in this range, v_p(i) >= k where p^k is current power
            # Actually, simpler view:
            # If i is a multiple of p, it gets one factor of p.
            # If i is a multiple of p^2, it gets another factor of p.
            # So for each power p^k, we multiply small_factor[multiple] by p.
            indices = np.arange(power, max_val + 1, power)
            small_factor[indices] *= p
            power *= p
    
    # Convert to list for faster access in the sliding window loop
    # Numpy scalar access in a loop can be slower than list access in CPython
    # But converting 1M ints to a list takes ~50ms, which is acceptable.
    sf = small_factor.tolist()
    
    # Sliding window initialization
    # Window for n=1000 covers sf[1001] to sf[1024]
    window_size = 24
    start_idx = 1000
    
    freq = {}
    
    # Initialize frequency map for the first window
    for i in range(1, window_size + 1):
        val = sf[start_idx + i]
        if val in freq:
            freq[val] += 1
        else:
            freq[val] = 1
            
    best_score = len(freq)
    best_n = start_idx
    
    # Slide the window
    # n goes from 1001 to max_n
    # Window for n covers sf[n+1] to sf[n+24]
    # Previous window (n-1) covered sf[n] to sf[n+23]
    # So we remove sf[n] and add sf[n+24]
    
    for n in range(start_idx + 1, max_n + 1):
        if time.time() > deadline:
            break
            
        # Remove the element leaving the window: sf[n]
        # The window for n-1 included sf[n] (which was the first element: (n-1)+1 = n)
        val_out = sf[n]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            del freq[val_out]
            
        # Add the new element entering the window: sf[n+24]
        # The window for n includes sf[n+24] (which is the last element: n+24)
        val_in = sf[n + 24]
        if val_in in freq:
            freq[val_in] += 1
        else:
            freq[val_in] = 1
            
        current_score = len(freq)
        if current_score < best_score:
            best_score = current_score
            best_n = n
            # Theoretical minimum is 1. If we find it, we can stop.
            if best_score == 1:
                break
                
    return best_n
