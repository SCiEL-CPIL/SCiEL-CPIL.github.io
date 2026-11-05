"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need to compute the "small-prime part" for each number m
    # i.e., the largest divisor of m composed only of primes < 24
    max_n = 1000000
    max_val = max_n + 24
    
    # Precompute small-prime part for each number up to max_val
    # small_part[m] = product of p^a for p in small_primes where p^a || m
    small_part = np.zeros(max_val + 1, dtype=np.int64)
    small_part[0] = 0
    small_part[1] = 1
    
    # For each number, compute its small-prime part
    # Use a sieve-like approach: for each small prime p, multiply into small_part
    # Actually, let's just compute directly for each m
    # But that's O(max_val * num_primes) which is fine
    
    # Better: use a modified sieve
    # Initialize small_part[m] = 1 for all m >= 1
    small_part[1:] = 1
    
    for p in small_primes:
        # For each multiple of p, divide out all factors of p
        # We need to track how many times p divides m
        # Approach: for each m, count exponent of p in m, then multiply small_part[m] by p^exp
        # This is expensive. Let's use a different approach.
        
        # For each m from 1 to max_val, compute the p-part
        # We can do this iteratively
        p_power = 1
        # For each m, the p-part is p^(v_p(m))
        # Let's compute v_p(m) for all m using a sieve
        vp = np.zeros(max_val + 1, dtype=np.int32)
        for multiple in range(p, max_val + 1, p):
            # Count how many times p divides multiple
            temp = multiple
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            vp[multiple] = count
        
        # Now multiply small_part[m] by p^vp[m]
        # p^vp[m] for all m
        p_powers = np.array([p ** v for v in vp], dtype=np.int64)
        small_part *= p_powers
    
    # Now small_part[m] is the complete factor of m composed of primes < 24
    # For m=0, it's 0 (we don't care)
    
    # Now slide a window of size 24 over small_part[1001:1000025]
    # For each window, count distinct values
    # Window starts at n, covers n+1 to n+24
    
    # We want to minimize the number of distinct small_part values in the window
    # 
    # Approach: for each possible starting index i (corresponding to n=i),
    # the window is small_part[i+1:i+25]
    # 
    # n ranges from 1000 to 1000000, so i ranges from 1000 to 1000000
    # Window is small_part[1001:1000025]
    
    # To count distinct values in a sliding window efficiently:
    # We can use a dictionary-based approach with a sliding window count
    
    start_idx = 1000  # n starts here, window is small_part[1001:1000025]
    end_idx = 1000000  # n ends here
    
    # small_part indices: for n, we look at small_part[n+1] through small_part[n+24]
    # So for n=1000: small_part[1001:10025]
    # For n=1000000: small_part[1000001:1000025]
    
    # Let's use a frequency dictionary and slide
    freq = {}
    distinct_count = 0
    
    # Initialize window for n=1000
    for i in range(1001, 10025):
        val = small_part[i]
        freq[val] = freq.get(val, 0) + 1
        if freq[val] == 1:
            distinct_count += 1
    
    best_n = 1000
    best_distinct = distinct_count
    
    # Slide the window
    for n in range(1001, end_idx + 1):
        # Remove small_part[n] (which was the first element of previous window at n-1)
        # Previous window for n-1: small_part[n] to small_part[n+23]
        # New window for n: small_part[n+1] to small_part[n+24]
        
        # Remove small_part[n]
        old_val = small_part[n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
            distinct_count -= 1
        
        # Add small_part[n+24]
        new_val = small_part[n + 24]
        freq[new_val] = freq.get(new_val, 0) + 1
        if freq[new_val] == 1:
            distinct_count += 1
        
        if distinct_count < best_distinct:
            best_distinct = distinct_count
            best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
