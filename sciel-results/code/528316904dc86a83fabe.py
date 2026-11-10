"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit

    # Precompute largest prime factor for all numbers up to N
    # Using a sieve: for each prime p, mark all multiples of p with lpf = p
    # Since we iterate p in increasing order, the last prime that divides n
    # (i.e., the largest one) will be the final value written.
    lpf = [1] * (N + 1)
    
    for p in range(2, N + 1):
        if lpf[p] == 1:  # p is prime (no smaller prime factor found)
            for j in range(p, N + 1, p):
                lpf[j] = p

    best_a = 1
    best_b = N - 1
    min_max_p = float('inf')
    
    # Iterate through all possible splits a + b = N
    # Since N is odd, one of a,b is even and one is odd.
    # We check all a from 1 to N-1.
    for a in range(1, N):
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        max_p = pa if pa > pb else pb
        
        if max_p < min_max_p:
            min_max_p = max_p
            best_a = a
            best_b = b
            
            # Theoretical lower bound:
            # If we find max_p = 2, that means both a and b are powers of 2 (or 1).
            # Since a+b=N (odd), one is even, one is odd.
            # If the odd one is 1, P(1)=1. If the even one is 2^k, P(2^k)=2.
            # So max(1, 2) = 2.
            # Can we get max_p = 1? Only if both are 1, so N=2. Here N=123457.
            # So 2 is the absolute minimum possible value for max(P(a), P(b)) 
            # IF there exists a power of 2, say 2^k, such that N - 2^k = 1.
            # i.e., 2^k = N - 1.
            # N - 1 = 123456. Is 123456 a power of 2?
            # 2^16 = 65536, 2^17 = 131072. No.
            # So the minimum is likely > 2.
            # However, if we find a solution with max_p=2, we can stop.
            if min_max_p == 2:
                break

    return [best_a, best_b]
# EVOLVE-BLOCK-END
