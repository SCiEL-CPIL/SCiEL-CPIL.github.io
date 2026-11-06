import time
import math
import numpy as np

def construct():
    """
    Returns an integer n in [10000, 1000000] that maximizes 
    powerful_part(n) * powerful_part(n+1).
    
    Structural change: Use a sieve-based approach to compute powerful parts.
    For each prime p and each exponent e>=2, identify numbers divisible by p^e 
    but not p^(e+1), and multiply their powerful_part by p^e.
    """
    start_time = time.time()
    DEADLINE = 82.0  # Safety margin for 90s limit
    
    N = 1000000
    MIN_N = 10000
    
    # 1. Build list of primes up to N using simple sieve
    is_prime = bytearray([1]) * (N + 1)
    is_prime[0] = 0
    if N >= 1:
        is_prime[1] = 0
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # 2. Compute powerful part using sieve approach
    # powerful_part[i] = product of p^e where e is the exact exponent of p in i, and e >= 2
    powerful_part = np.ones(N + 1, dtype=np.int64)
    
    for p in primes:
        # For this prime p, find all exponents e >= 2 such that p^e <= N
        p_power = p * p  # p^2
        e = 2
        while p_power <= N:
            # Numbers divisible by p^e but not p^(e+1)
            # First, mark multiples of p^e
            # Then, unmark multiples of p^(e+1) if e+1 power exists
            
            # All multiples of p^e
            multiples = np.arange(p_power, N + 1, p_power)
            
            # If p^(e+1) <= N, remove multiples of p^(e+1)
            next_power = p_power * p
            if next_power <= N:
                # Multiples of p^(e+1) are a subset of multiples of p^e
                # We want numbers divisible by p^e but NOT p^(e+1)
                # So: multiples of p^e minus multiples of p^(e+1)
                next_multiples = np.arange(next_power, N + 1, next_power)
                # Create a boolean mask
                mask = np.ones(len(multiples), dtype=bool)
                # For each multiple of next_power, check if it's in multiples
                # More efficient: use set difference or boolean indexing
                # Since next_multiples are every p-th element of multiples
                # multiples = [p^2, 2*p^2, 3*p^2, ...]
                # next_multiples = [p^3, 2*p^3, 3*p^3, ...] = [p*p^2, p*2*p^2, ...]
                # So next_multiples[i] = multiples[i*p]
                # Thus, indices to exclude: 0, p, 2p, 3p, ...
                exclude_indices = np.arange(0, len(multiples), p)
                mask[exclude_indices] = False
                multiples = multiples[mask]
            
            if len(multiples) > 0:
                powerful_part[multiples] *= p_power
            
            # Time check
            if time.time() - start_time > DEADLINE:
                break
            p_power = next_power
            e += 1
        if time.time() - start_time > DEADLINE:
            break
    
    # 3. Find n that maximizes powerful_part[n] * powerful_part[n+1]
    # Use numpy for vectorized scoring
    if time.time() - start_time > DEADLINE:
        # Return a reasonable default if we timed out during computation
        return MIN_N
    
    max_n_to_check = N - 1
    scores = powerful_part[MIN_N:max_n_to_check + 1] * powerful_part[MIN_N + 1:max_n_to_check + 2]
    best_idx = np.argmax(scores)
    best_n = MIN_N + best_idx
    
    return int(best_n)
