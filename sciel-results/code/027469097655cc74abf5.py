import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 1000000
    
    # Initialize LPF array
    # lpf[i] will hold the largest prime factor of i
    # Initialize with 0 to indicate "not set" or handle 0/1 separately
    lpf = np.zeros(N + 2, dtype=np.int32)
    
    # Sieve for largest prime factor
    # We iterate p from 2 to N+1
    # If lpf[p] is 0, p is prime (since no smaller prime divided it)
    # Then we update all multiples of p
    # Note: We want the LARGEST prime factor.
    # If we iterate p in increasing order and overwrite lpf[multiple] = p,
    # the last prime that divides 'multiple' (the largest one) will remain.
    
    # Optimization: Only iterate up to N+1.
    # Using a standard loop for primes is fine, but we can optimize the inner update.
    
    # A more efficient way in NumPy is to identify primes first?
    # Actually, the "overwrite" method works:
    # for p in range(2, N + 2):
    #     if lpf[p] == 0: # p is prime
    #         lpf[p::p] = p
    #
    # This is O(N log log N) operations, but vectorized.
    # Let's verify this logic.
    # If i = 12. Primes dividing 12 are 2, 3.
    # p=2: lpf[2], lpf[4], lpf[6], lpf[8], lpf[10], lpf[12]... become 2.
    # p=3: lpf[3], lpf[6], lpf[9], lpf[12]... become 3.
    # Final lpf[12] is 3. Correct.
    
    for p in range(2, N + 2):
        if lpf[p] == 0:  # p is prime
            # Update all multiples of p
            # This is a strided slice assignment
            lpf[p::p] = p
            
        # Optional: Time check, though this should be very fast
        if time.time() > deadline:
            break

    # Ensure we have valid LPF for all numbers in range
    # 1 has no prime factors, but the problem asks for n >= 10000, so 1 is irrelevant.
    # 0 is irrelevant.
    
    # We need n in [10000, 1000000]
    # We want to minimize max(lpf[n], lpf[n+1])
    
    # Create array of n values
    # n ranges from 10000 to N
    # n+1 ranges from 10001 to N+1
    
    n_start = 10000
    n_end = N
    
    # Extract relevant slices
    # lpf_n = lpf[n_start : n_end + 1]
    # lpf_np1 = lpf[n_start + 1 : n_end + 2]
    
    lpf_n = lpf[n_start : n_end + 1]
    lpf_np1 = lpf[n_start + 1 : n_end + 2]
    
    # Compute max for each pair
    pair_max = np.maximum(lpf_n, lpf_np1)
    
    # Find the index of the minimum value
    min_idx = np.argmin(pair_max)
    
    best_n = n_start + min_idx
    best_val = pair_max[min_idx]
    
    return int(best_n)
