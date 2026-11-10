import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (evaluator kills at 90s)
    
    MAX_N = 1000000
    WINDOW = 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need smooth parts for numbers in range [1001, 1000024]
    # (since n is in [1000, 1000000] and we look at n+1 to n+24)
    min_val = 1001
    max_val = MAX_N + WINDOW  # 1000024
    limit = max_val + 1
    
    # Compute smooth part using a sieve-like approach
    # smooth[m] = product of p^e for each small prime p where p^e || m
    # Use numpy for speed
    
    # For each small prime, we need to compute the exponent of p in each number
    # then multiply p^exponent into the result
    
    # Approach: for each prime p, for each power p^k, mark multiples
    # but we need to be careful not to overcount
    
    # Better: for each number m, factor out small primes
    # But that's slow in Python. Let's use a sieve approach.
    
    # For each small prime p:
    #   For each multiple m of p, count how many times p divides m
    #   Then smooth[m] *= p^count
    #
    # We can do this with numpy by processing each prime separately.
    
    # Create array of indices
    # smooth_part[m] for m in [min_val, max_val]
    # Use a dictionary or array indexed by m - min_val
    
    # Actually, let's use a full array up to max_val for simplicity
    # max_val ~ 1000024, so array of size ~1M is fine
    
    smooth = np.ones(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        # For each multiple of p, we need to divide out all factors of p
        # and multiply by p^count
        # 
        # More efficiently: for each power p^k, for each multiple of p^k
        # that is NOT a multiple of p^(k+1), add one to the exponent count
        # 
        # Actually simpler: for each m that is a multiple of p,
        # count = 0
        # while m % p == 0: count++, m //= p
        # smooth[original_m] *= p^count
        #
        # In numpy, we can do this by iterating over powers of p:
        # For p^1: all multiples of p get at least one factor
        # For p^2: all multiples of p^2 get at least two factors
        # etc.
        # So exponent of p in m = number of powers p^k that divide m
        # = sum over k of [p^k divides m]
        #
        # So smooth[m] *= p^(number of k such that p^k | m)
        # = product over k of p if p^k | m
        # = product over k of (p if m % p^k == 0 else 1)
        
        power = p
        while power <= max_val:
            # All multiples of 'power' get multiplied by p
            multiples = np.arange(power, max_val + 1, power)
            smooth[multiples] *= p
            power *= p
            if power > max_val:
                break
    
    # Now smooth[m] is the complete small-prime factor of m
    # Verify with a few examples:
    # m=12=2^2*3: p=2: 12%2==0 -> *2, 12%4==0 -> *2, 12%8!=0 -> stop. So *4 from p=2
    # p=3: 12%3==0 -> *3, 12%9!=0 -> stop. So *3 from p=3. Total: 4*3=12. Correct!
    
    # Now extract the window values
    # We need smooth[m] for m in [min_val, max_val]
    # Convert to list for easier manipulation, or keep as numpy
    
    # For the sliding window, we need to track distinct values
    # Since smooth values can be large, let's use a dictionary-based approach
    # but with numpy for the initial extraction
    
    # Extract relevant portion
    smooth_arr = smooth[min_val:max_val + 1]  # length = max_val - min_val + 1 = 1000024 - 1001 + 1 = 999024
    
    # Now slide window of size 24 over smooth_arr
    # Window starting at position i covers smooth_arr[i:i+24]
    # n = 1000 + i (since smooth_arr[0] = smooth[1001] = smooth[n+1] for n=1000)
    # Wait: n=1000 -> numbers 1001..1024 -> smooth_arr[0:24]
    # n=1001 -> numbers 1002..1025 -> smooth_arr[1:25]
    # So n = 1000 + i where i is the start index in smooth_arr
    
    # Use a frequency counter with numpy
    # For efficiency, let's use a dictionary
    
    from collections import defaultdict
    
    best_n = 1000
    best_count = WINDOW + 1
    
    # Initial window: i=0, covers smooth_arr[0:24]
    freq = defaultdict(int)
    for j in range(WINDOW):
        val = int(smooth_arr[j])
        freq[val] += 1
    
    best_count = len(freq)
    best_n = 1000
    
    # Slide the window
    n_max = MAX_N  # 1000000
    # i ranges from 0 to n_max - 1000 = 999000
    # smooth_arr has length 999024, so last valid start is 999024 - 24 = 999000. Good.
    
    for i in range(1, 999001):
        # Remove element at position i-1
        old_val = int(smooth_arr[i - 1])
        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
        
        # Add element at position i + WINDOW - 1
        new_idx = i + WINDOW - 1
        new_val = int(smooth_arr[new_idx])
        freq[new_val] += 1
        
        cur_count = len(freq)
        if cur_count < best_count:
            best_count = cur_count
            best_n = 1000 + i
        
        # Check time
        if (i & 0xFFFF) == 0:  # every 65536 iterations
            if time.time() > deadline:
                break
    
    return best_n
