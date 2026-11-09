import time
import sys
import numpy as np

def construct():
    start = time.time()
    deadline = start + 85  # Safety margin for 90s limit
    
    N = 1000000
    WINDOW = 24
    MAX_VAL = N + WINDOW
    
    # Primes strictly below 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # For each m, S[m] is the product of all prime powers p^k where p < 24 and p^k divides m
    # Equivalently, S[m] = m / (m with all prime factors >= 24 removed)
    # Or: S[m] = product over p in PRIMES of (p ^ (exponent of p in m))
    
    # Compute S using a sieve-like approach
    S = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in PRIMES:
        # For each power of p, multiply S[m] by that power for all m divisible by it
        power = p
        while power <= MAX_VAL:
            # All multiples of power get multiplied by p (one more factor)
            # But we need to be careful: if m is divisible by p^k, it gets p^k total
            # Better approach: for each m, divide out all factors of p, then multiply back
            pass
            power *= p
    
    # Alternative: compute S by dividing m by all prime factors >= 24
    # S[m] = m / (largest divisor of m with no prime factors < 24)
    
    # Let's use the spf approach but vectorize the S computation
    spf = np.arange(MAX_VAL + 1, dtype=np.int32)
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:
            spf[i*i::i] = np.minimum(spf[i*i::i], i)
    
    S = np.ones(MAX_VAL + 1, dtype=np.int64)
    for m in range(2, MAX_VAL + 1):
        p = spf[m]
        if p < 24:
            S[m] = S[m // p] * p
        else:
            S[m] = S[m // p]
    
    # Now do the sliding window
    freq = {}
    distinct = 0
    
    def add_val(v):
        nonlocal distinct
        if v in freq:
            freq[v] += 1
        else:
            freq[v] = 1
            distinct += 1
            
    def remove_val(v):
        nonlocal distinct
        freq[v] -= 1
        if freq[v] == 0:
            del freq[v]
            distinct -= 1
            
    for i in range(1001, 1025):
        add_val(int(S[i]))
        
    best_n = 1000
    best_distinct = distinct
    
    for n in range(1001, N + 1):
        if time.time() > deadline - 2:
            break
        remove_val(int(S[n]))
        add_val(int(S[n+24]))
        
        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n
            if best_distinct <= 2:
                pass
                
    return best_n
