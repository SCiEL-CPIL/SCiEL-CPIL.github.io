"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    N = 1000000
    # Powerful part P(n) = product of p^e for all prime powers p^e || n with e >= 2
    # i.e., for each prime p dividing n, if p^e || n, then:
    #   if e == 1: contribute 1 (squarefree part)
    #   if e >= 2: contribute p^e (full prime power)
    
    # We want to compute P(n) for all n in [1, N] efficiently.
    
    # Strategy: sieve-based approach
    # Initialize P[n] = 1 for all n
    # For each prime p, for each power p^k with k >= 2:
    #   For each multiple m of p^k where p does NOT divide m/p^k (i.e., p^(k) || m exactly):
    #     Actually, we need to be more careful.
    
    # Let's think differently. For a number n = prod p_i^(e_i):
    # P(n) = prod p_i^(e_i) for those i with e_i >= 2
    # 
    # Alternative: P(n) = n * prod_{p|n, e_p=1} (1/p)
    # Wait, that's not right either. Let me re-derive.
    # 
    # n = prod p^(e_p)
    # P(n) = prod_{p: e_p >= 2} p^(e_p)
    # 
    # So P(n) = n / prod_{p: e_p = 1} p
    # 
    # That's much simpler! P(n) = n divided by the product of primes that appear to the first power only.
    
    # So we need for each n, the product of primes p such that p || n (exactly divides, e_p = 1).
    
    # Let S(n) = product of primes p such that p || n (exactly, e_p = 1)
    # Then P(n) = n / S(n)
    
    # We can compute S(n) using a sieve:
    # Initialize S[n] = 1 for all n
    # For each prime p:
    #   For each multiple m of p:
    #     If p^2 does not divide m, then multiply S[m] by p
    
    # This is O(N log log N) with proper implementation.
    
    S = [1] * (N + 1)
    
    # Sieve of Eratosthenes to find primes and process
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    for p in primes:
        p2 = p * p
        # For multiples of p, check if p^2 divides them
        for m in range(p, N + 1, p):
            if m % p2 != 0:
                S[m] *= p
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if time.time() > deadline:
        # Fallback: return a reasonable value
        return 10000
    
    # Now P(n) = n // S[n] for n where S[n] divides n (which it should)
    # Compute objective: P(n) * P(n+1) for n in [10000, 999999]
    
    best_n = 10000
    best_val = 0
    
    for n in range(10000, N):
        Pn = n // S[n]
        Pn1 = (n + 1) // S[n + 1]
        val = Pn * Pn1
        if val > best_val:
            best_val = val
            best_n = n
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
