"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n*(n+1).
    
    Strategy:
    1. Precompute smallest prime factor (SPF) up to 1000001 using a sieve.
    2. For each n in range, compute LPF(n) and LPF(n+1) in O(log n) using SPF.
    3. The largest prime factor of n*(n+1) is max(LPF(n), LPF(n+1)).
    4. Track the minimum and return the corresponding n.
    
    Time complexity: O(N log log N) for sieve + O(N log N) for factorization.
    With N=10^6, this completes well within the 60s limit.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin: 55s out of 60s
    
    N = 1000000
    # We need SPF up to N+1 = 1000001
    max_val = N + 1
    
    # Sieve for smallest prime factor
    spf = list(range(max_val + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def lpf_via_spf(x):
        """Compute largest prime factor of x using SPF table."""
        if x <= 1:
            return 1
        max_pf = 1
        while x > 1:
            p = spf[x]
            max_pf = max(max_pf, p)
            while x % p == 0:
                x //= p
        return max_pf
    
    best_n = 10000
    best_lpf = float('inf')
    
    for n in range(10000, N + 1):
        # Check deadline
        if time.time() > deadline:
            break
        
        lpf_n = lpf_via_spf(n)
        lpf_np1 = lpf_via_spf(n + 1)
        lpf_product = max(lpf_n, lpf_np1)
        
        if lpf_product < best_lpf:
            best_lpf = lpf_product
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
