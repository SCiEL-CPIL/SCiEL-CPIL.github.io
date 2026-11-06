import time
import math
import numpy as np

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    LO = 10000
    HI = 1000000
    MAX_VAL = HI + 1  # 1000001
    
    # Sieve to find primes and compute largest prime factor for all numbers up to MAX_VAL
    # Using a modified sieve: for each prime p, mark all multiples with lpf = p
    # Since we go in increasing order of p, the final lpf[m] is the largest prime factor
    
    # We need primes up to MAX_VAL
    # Use a bytearray for is_prime for memory efficiency, then convert to numpy for speed
    is_prime = bytearray([1]) * (MAX_VAL + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    
    limit = int(math.isqrt(MAX_VAL))
    for i in range(2, limit + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    
    # Now compute lpf using NumPy for speed
    # lpf[m] will be the largest prime factor of m
    lpf = np.zeros(MAX_VAL + 1, dtype=np.int32)
    
    # Iterate through primes in increasing order
    # For each prime p, mark all multiples with p
    # The last prime to mark a number m is its largest prime factor
    for p in range(2, MAX_VAL + 1):
        if is_prime[p]:
            lpf[p::p] = p
    
    # Find n in [LO, HI] that minimizes max(lpf[n], lpf[n+1])
    # Vectorized approach
    # n ranges from LO to HI
    # lpf[n] corresponds to indices LO to HI
    # lpf[n+1] corresponds to indices LO+1 to HI+1
    
    lpf_n = lpf[LO:HI+1]
    lpf_np1 = lpf[LO+1:HI+2]
    
    vals = np.maximum(lpf_n, lpf_np1)
    
    # Find the index of the minimum value
    min_idx = np.argmin(vals)
    
    # The index in vals corresponds to n = LO + min_idx
    # Ensure the result is a standard Python integer
    best_n = int(LO + min_idx)
    
    return best_n
