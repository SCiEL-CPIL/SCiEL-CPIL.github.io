import math

def construct():
    N = 123457
    # Precompute largest prime factor for all numbers 1..N using a sieve
    lp = [0] * (N + 1)  # lp[i] = largest prime factor of i
    lp[1] = 1
    
    # Sieve: for each prime p, mark all multiples with lp = p
    # We need primes up to N
    # Use a standard sieve to find primes, then assign largest prime factor
    
    # Actually, more efficient: initialize lp[i] = 0, then for each i from 2 to N,
    # if lp[i] == 0, i is prime, and for all multiples j of i, set lp[j] = i
    # Since we go in increasing order, the last prime that divides j will be the largest.
    
    for i in range(2, N + 1):
        if lp[i] == 0:
            # i is prime
            for j in range(i, N + 1, i):
                lp[j] = i
    
    # Now lp[n] is the largest prime factor of n for all n >= 2
    # lp[1] = 1
    
    # Scan all decompositions
    best_max = N  # worst case: one of them is N (prime), so max prime factor is N
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N):
        b = N - a
        pa = lp[a] if a >= 2 else 1
        pb = lp[b] if b >= 2 else 1
        m = pa if pa > pb else pb
        if m < best_max:
            best_max = m
            best_a = a
            best_b = b
        # Early termination: if best_max == 2, that's the absolute minimum possible
        # (since both must be >= 1, and the smallest prime factor is 2)
        # Actually the minimum possible max prime factor is at least 2 (unless one is 1)
        # If one is 1, P(1)=1, so max could be P(N-1). 
        # The absolute minimum possible is 1 if both are 1, but 1+1=2 != 123457.
        # So minimum possible is at least 2.
        if best_max == 2:
            break
    
    return [best_a, best_b]
