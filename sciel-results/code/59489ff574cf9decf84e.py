import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin

    lower = 1000
    upper = 1000000
    max_val = upper + 24
    N = max_val - lower + 1
    
    # We need smooth parts for numbers from lower+1 to max_val
    # Indices in our array will be 1 to N-1 if we start at lower
    # Let's create an array of numbers from lower+1 to max_val
    # length = max_val - (lower+1) + 1 = max_val - lower = 1000000 + 24 - 1000 = 999024
    # Wait, n ranges from 1000 to 1000000.
    # For n=1000, we look at 1001..1024.
    # For n=1000000, we look at 1000001..1000024.
    # So we need smooth parts for [1001, 1000024].
    
    start_num = lower + 1
    end_num = max_val
    count = end_num - start_num + 1
    nums = np.arange(start_num, end_num + 1, dtype=np.int64)
    
    smooth = np.ones(count, dtype=np.int64)
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    for p in small_primes:
        temp = nums.copy()
        factor = np.ones(count, dtype=np.int64)
        while True:
            mask = (temp % p) == 0
            if not np.any(mask):
                break
            temp[mask] //= p
            factor[mask] *= p
        smooth *= factor
        
    # Now smooth[i] is the smooth part of (start_num + i)
    # We slide a window of size 24 over smooth.
    # Window for n corresponds to indices (n - start_num) to (n - start_num + 23)
    # n starts at lower (1000).
    # Index for n=1000 is 0. Window: smooth[0:24].
    # Index for n=1000000 is 1000000 - 1001 = 998999. Window: smooth[998999:998999+24].
    
    best_n = lower
    best_count = 25  # Max possible is 24, so 25 is infinity for our purposes
    
    freq = {}
    distinct = 0
    
    # Initialize window for n = lower
    for i in range(24):
        v = int(smooth[i])
        if v not in freq:
            freq[v] = 0
            distinct += 1
        freq[v] += 1
        
    if distinct < best_count:
        best_count = distinct
        best_n = lower
        
    # Slide window
    # n goes from lower+1 to upper
    # The index in smooth for n is idx = n - start_num
    # The window is smooth[idx : idx+24]
    # Previous window was smooth[idx-1 : idx-1+24]
    # Out: smooth[idx-1]
    # In: smooth[idx+23]
    
    for n in range(lower + 1, upper + 1):
        if time.time() > deadline:
            break
            
        idx = n - start_num
        
        # Remove smooth[idx] (which corresponds to the first element of the previous window)
        v_out = int(smooth[idx])
        if v_out in freq:
            freq[v_out] -= 1
            if freq[v_out] == 0:
                del freq[v_out]
                distinct -= 1
        else:
            # This should not happen if logic is correct, but prevents crash
            pass

        # Add smooth[idx+24] (which corresponds to the last element of the current window)
        v_in = int(smooth[idx + 24])
        if v_in not in freq:
            freq[v_in] = 0
            distinct += 1
        freq[v_in] += 1
        
        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count == 1: # Theoretical minimum
                break
                
    return best_n
