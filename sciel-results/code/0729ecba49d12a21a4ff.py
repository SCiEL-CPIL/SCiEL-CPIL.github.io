"""Program to find n in [1000, 1000000] minimizing distinct small-prime factors in window."""

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Primes strictly below 24: 2, 3, 5, 7, 11, 13, 17, 19, 23
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    n_primes = len(small_primes)
    window_size = 24
    
    # We need to compute for each n in [1000, 1000000], 
    # for each i in 1..24, the factor of n+i composed of primes in small_primes
    # Then count distinct such factors across the window
    
    # Strategy: Precompute the "small-prime factor" for all numbers up to 1000000+24
    # For each number m, extract the part composed of primes < 24
    # This is: m with all factors of primes >= 27 removed
    
    max_val = 1000000 + 24
    n_max = max_val + 1
    
    # Array to store the small-prime factor for each number
    small_factor = np.ones(n_max, dtype=np.int64)
    
    # For each small prime p, multiply into small_factor wherever p divides the number
    # Actually, we need to extract the COMPLETE factor composed of small primes
    # So for number m, small_factor[m] = product of p^e where p is small prime and p^e || m
    
    # Let's compute this by dividing out large primes
    # Alternative: for each small prime p, repeatedly divide numbers divisible by p
    
    # More efficient: use a sieve-like approach
    # small_factor[m] starts at 1, and we multiply by p for each power of p dividing m
    
    # Actually, let's think differently. For each m, we want the largest divisor d of m
    # such that all prime factors of d are in small_primes.
    
    # This equals: m / (m with all small prime factors removed)
    # Or equivalently: the part of m that is "smooth" with respect to small primes
    
    # Let's compute it directly:
    # For each m from 2 to max_val, compute small_factor[m]
    
    # Efficient approach:
    # 1. Start with small_factor = arange(1, n_max)
    # 2. For each large prime q >= 27, divide out all factors of q from small_factor
    # But this is complex. Let's do it the other way.
    
    # Better: For each number m, compute by trial division by small primes
    # But that's O(n * n_primes * log(max_val)) which might be too slow in Python
    
    # Let's use numpy for vectorized operations:
    # small_factor[m] = m with all factors of primes >= 27 removed
    
    # Alternative: Compute the "co-prime to small primes" part
    # coprime_part[m] = m / small_factor[m] where all prime factors are >= 27
    
    # Let's just do: for each small prime p, extract all powers of p
    # small_factor[m] = product over p in small_primes of p^(v_p(m))
    
    # We can compute this as:
    # Start with result = ones
    # For each p in small_primes:
    #   For each power p^k: divide out p^k from numbers still divisible by p^k
    
    # Actually simpler: 
    # temp = arange(1, n_max)  # represents the remaining part
    # small_part = ones
    # For each p in small_primes:
    #   while any number in temp is divisible by p:
    #     mask = temp % p == 0
    #     small_part[mask] *= p
    #     temp[mask] //= p
    
    # This is O(sum over p of log_p(max_val) * n) which is fine
    
    temp = np.arange(1, n_max, dtype=np.int64)
    small_part = np.ones(n_max, dtype=np.int64)
    
    for p in small_primes:
        # Keep dividing by p while divisible
        while True:
            mask = (temp % p == 0) & (temp > 0)
            if not np.any(mask):
                break
            small_part[mask] *= p
            temp[mask] //= p
            elapsed = time.time() - start_time
            if elapsed > deadline - 5:
                break
    
    # Now small_part[m] is the factor of m composed entirely of primes < 24
    # For m=1, small_part[1]=1
    
    # For each n in [1000, 1000000], compute distinct count of small_part[n+1..n+24]
    # Use a sliding window approach
    
    # Since we have up to 9 distinct small primes, the small_part values are of the form
    # 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17^g * 19^h * 23^i
    # We can encode these as bitmasks or use the value itself for distinct counting
    
    # For the window of 24 consecutive numbers, we need to count distinct small_part values
    
    # Let's precompute for all n in [1000, 1000000]:
    # distinct_count[n] = number of distinct values in small_part[n+1 : n+25]
    
    # This is a sliding window distinct count problem
    # We can use a frequency array and sliding window
    
    # Map small_part values to indices for frequency counting
    # The small_part values can be large, but there are limited distinct values
    # Let's find unique values
    
    unique_vals = np.unique(small_part[1001:1000000+25])
    val_to_idx = {v: i for i, v in enumerate(unique_vals)}
    n_unique = len(unique_vals)
    
    # For each position m in [1001, 1000000+24], get the index
    # indices[m-1001] = val_to_idx[small_part[m]]
    
    # Actually, let's work with positions 0 to 1000000-1 where position i corresponds to n=i+1000
    # and we look at small_part[i+1001 : i+1000+25]
    
    n_candidates = 1000000 - 1000 + 1  # 999001 candidates
    
    # Create index array for the range [1001, 1000000+24]
    idx_arr = np.empty(n_candidates + 24, dtype=np.int32)
    for i in range(len(unique_vals)):
        pass
    
    # This mapping might be slow. Let's use a different approach.
    # Since small_part values are products of powers of 9 primes, and max value is <= 1000024,
    # we can just use the value directly in a dictionary for the sliding window.
    
    # Sliding window with a Counter-like structure
    # Use a list to track frequencies
    
    best_n = 1000
    best_count = 25  # Max possible is 24 (window size), but we'll initialize high
    
    # Initialize window for n=1000: positions 1001 to 1024
    freq = {}
    for i in range(1, 25):
        v = int(small_part[1000 + i])
        freq[v] = freq.get(v, 0) + 1
    
    best_count = len(freq)
    best_n = 1000
    
    # Slide the window
    for n in range(1001, 1000001):
        # Remove position n-1+1 = n (which was n-1000+1000+1... let me re-think)
        # Window for n is [n+1, n+24]
        # Window for n-1 is [n, n+23]
        # So remove n, add n+24
        
        remove_val = int(small_part[n])
        add_val = int(small_part[n + 24])
        
        freq[remove_val] = freq.get(remove_val, 0) - 1
        if freq[remove_val] == 0:
            del freq[remove_val]
        
        freq[add_val] = freq.get(add_val, 0) + 1
        
        current_distinct = len(freq)
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 2:
            break
    
    return best_n
# EVOLVE-BLOCK-END
