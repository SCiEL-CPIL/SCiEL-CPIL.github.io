"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find two positive integers a, b with a + b = 123457
    that minimize max(P(a), P(b)), where P(n) is the largest prime factor.
    
    Strategy:
    1. Precompute largest prime factor (LPF) for all numbers up to 123457 using a sieve.
    2. Scan all valid pairs (a, b) = (i, N-i) for i in 1..N-1.
    3. Track the pair that minimizes max(LPF[i], LPF[N-i]).
    """
    N = 123457
    deadline = time.time() + 50  # Safety margin before 60s limit
    
    # Step 1: Sieve to compute largest prime factor for all numbers up to N
    lpf = list(range(N + 1))
    
    # Sieve of Eratosthenes variant: for each prime p, update lpf for multiples
    for p in range(2, N + 1):
        if lpf[p] == p:  # p is prime
            for multiple in range(p, N + 1, p):
                lpf[multiple] = p
    
    # Step 2: Scan all pairs to find the minimum of max(LPF[a], LPF[b])
    best_max = float('inf')
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N):
        b = N - a
        # Check deadline
        if time.time() > deadline:
            break
        pa = lpf[a]
        pb = lpf[b]
        current_max = pa if pa > pb else pb
        if current_max < best_max:
            best_max = current_max
            best_a = a
            best_b = b
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
