"""Find n in [10000, 1000000] minimizing the largest prime factor of n(n+1)."""
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N_LOW = 10000
    N_HIGH = 1000000
    
    # We need the largest prime factor of n(n+1) for n in [N_LOW, N_HIGH]
    # This is max(lpf(n), lpf(n+1)) where lpf(x) is the largest prime factor of x.
    #
    # Strategy:
    # 1. Sieve smallest prime factor (SPF) up to N_HIGH + 1.
    # 2. For each number up to N_HIGH+1, compute its largest prime factor using SPF.
    # 3. For each n in [N_LOW, N_HIGH], compute max(lpf[n], lpf[n+1]).
    # 4. Find the n that minimizes this value.
    
    max_val = N_HIGH + 1  # we need lpf up to N_HIGH + 1
    
    # Build smallest prime factor sieve
    spf = np.arange(max_val + 1, dtype=np.int64)
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute largest prime factor for each number using SPF
    # lpf[x] = largest prime factor of x
    lpf = np.zeros(max_val + 1, dtype=np.int64)
    lpf[0] = 0
    lpf[1] = 1
    for x in range(2, max_val + 1):
        # Decompose x using SPF to find largest prime factor
        remaining = x
        lpf_val = 1
        while remaining > 1:
            p = spf[remaining]
            lpf_val = max(lpf_val, p)
            while remaining % p == 0:
                remaining //= p
        lpf[x] = lpf_val
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: return a reasonable value
        return N_LOW
    
    # For each n in [N_LOW, N_HIGH], score = max(lpf[n], lpf[n+1])
    n_range = np.arange(N_LOW, N_HIGH + 1)
    scores = np.maximum(lpf[n_range], lpf[n_range + 1])
    
    # Find the minimum score and corresponding n
    min_score = np.min(scores)
    # Get all n that achieve the minimum
    candidates = n_range[scores == min_score]
    best_n = int(candidates[0])
    
    return best_n
