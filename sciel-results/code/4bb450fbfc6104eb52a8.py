"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _largest_prime_factor(n, spf):
    if n <= 1:
        return 1
    max_pf = 1
    while n > 1:
        p = spf[n]
        max_pf = max(max_pf, p)
        while n % p == 0:
            n //= p
    return max_pf

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s margin
    
    N = 123457
    
    # Build smallest prime factor table up to N
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # For each a, compute P(a) and P(N-a), track best
    best_max_pf = float('inf')
    best_a = 1
    
    for a in range(1, N):
        if time.time() > deadline:
            break
        b = N - a
        pa = _largest_prime_factor(a, spf)
        pb = _largest_prime_factor(b, spf)
        m = max(pa, pb)
        if m < best_max_pf:
            best_max_pf = m
            best_a = a
            if best_max_pf == 2:  # Can't do better than 2 (both powers of 2, but sum is odd)
                break
    
    return [best_a, N - best_a]
# EVOLVE-BLOCK-END
