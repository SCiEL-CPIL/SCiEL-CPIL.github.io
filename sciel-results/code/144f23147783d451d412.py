"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import Counter

def construct():
    start = time.time()
    deadline = start + 80  # safety margin before 90s limit

    N = 1000000
    WINDOW = 24
    MAX_VAL = N + WINDOW
    
    # Primes strictly below 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    P_PROD = 1
    for p in PRIMES:
        P_PROD *= p

    # Step 1: Check for distinct count = 1
    # This requires s[n+i] = 1 for all i in 1..24
    # i.e., n+i is coprime to P = prod(PRIMES)
    
    # is_coprime[m] is True if m is coprime to P
    is_coprime = np.ones(MAX_VAL + 1, dtype=bool)
    indices = np.arange(MAX_VAL + 1)
    for p in PRIMES:
        is_coprime[indices % p == 0] = False
        
    # Sliding window sum of is_coprime
    # We want window of size 24 to be all True (sum == 24)
    # Window covers indices n+1 to n+24
    prefix_sum = np.cumsum(is_coprime)
    
    # i ranges from 1001 to MAX_VAL - 23
    # sum(is_coprime[i : i+24]) == 24
    # prefix_sum[i+23] - prefix_sum[i-1] == 24
    
    left = np.arange(1000, MAX_VAL - 23) # i-1
    right = np.arange(1024, MAX_VAL + 1) # i+23
    
    window_sums = prefix_sum[right] - prefix_sum[left]
    
    matches = np.where(window_sums == 24)[0]
    if len(matches) > 0:
        k = matches[0]
        i = 1001 + k
        n = i - 1
        return n

    # Step 2: General minimization
    # Compute s[m] = product of p^v_p(m) for p in PRIMES
    s = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in PRIMES:
        k = 1
        power = p
        while power <= MAX_VAL:
            next_power = power * p
            mask = (indices % power == 0)
            if next_power <= MAX_VAL:
                mask &= (indices % next_power != 0)
            s[mask] *= power
            power = next_power
            k += 1

    # We want to minimize the number of distinct values in s[n+1 : n+25]
    
    # Since s[m] values are large, we can't use a simple array.
    # But the number of distinct values in any window of size 24 is at most 24.
    
    best_n = 1000
    best_distinct = 25
    
    # Initialize window for n=1000
    # Values are s[1001] to s[1024]
    freq = Counter()
    for i in range(1001, 1025):
        freq[int(s[i])] += 1
        
    distinct = len(freq)
    best_distinct = distinct
    best_n = 1000

    # Slide window
    # n goes from 1001 to N
    # Remove s[n], add s[n+24]
    
    # Optimization: if best_distinct is 2, we can stop if we don't find 1 (already checked)
    # But we should check if 1 is possible. We already did.
    # So if best_distinct == 2, we can return immediately?
    # The task asks to MINIMIZE. If 1 is impossible, 2 is the minimum possible.
    # So if we find 2, we can return.
    
    if best_distinct == 2:
        return best_n

    for n in range(1001, N + 1):
        if time.time() > deadline:
            break
            
        # Remove s[n]
        v_out = int(s[n])
        freq[v_out] -= 1
        if freq[v_out] == 0:
            del freq[v_out]
            
        # Add s[n+24]
        v_in = int(s[n + 24])
        freq[v_in] += 1
        
        distinct = len(freq)
        
        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n
            if best_distinct == 2:
                return best_n

    return best_n

# EVOLVE-BLOCK-END
