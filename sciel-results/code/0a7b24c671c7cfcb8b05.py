"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import isqrt

def construct():
    start_time = time.time()
    DEADLINE = start_time + 78  # safety margin before 90s kill
    N = 1000000
    MIN_N = 10000

    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = list(range(N + 1))
    for i in range(2, isqrt(N) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # 2. Multiplicative number-theoretic sieve to compute powerful part P(n) for all n in [1, N]
    #    P(n) = product of p^e for each prime p where e = v_p(n) >= 2
    #    Initialize P(n) = 1 for all n.
    #    For each prime p, for each k >= 2 with p^k <= N, multiply p^k into P(m) for every multiple m of p^k.
    #    Wait, that's not quite right. If n has p^3, then p^2 divides n and p^3 divides n.
    #    We want to multiply p^3 (the full prime power) into P(n), not p^2 * p^3.
    #    Correct approach: For each prime p, for each k >= 2, let pk = p^k.
    #    For multiples m of pk that are NOT multiples of p^(k+1), multiply p^k into P(m).
    #    This ensures each prime contributes exactly its full prime power to P(m).
    
    P = [1] * (N + 1)
    
    # Find all primes up to N
    primes = []
    for i in range(2, N + 1):
        if spf[i] == i:
            primes.append(i)
    
    for p in primes:
        pk = p * p
        while pk <= N:
            p_next = pk * p
            # For multiples of pk: m = pk, 2*pk, 3*pk, ...
            # But we only want those where v_p(m) is exactly k, i.e., m is a multiple of pk but not p_next
            # Actually, simpler: for each multiple m of pk, if m is not a multiple of p_next,
            # then v_p(m) = k, so we multiply p^k = pk into P[m].
            # But if m IS a multiple of p_next, then v_p(m) > k, and we'll handle it in the next iteration.
            # So: for m in range(pk, N+1, pk):
            #     if m % p_next != 0:
            #         P[m] *= pk
            # This is correct but the modulo check is expensive. Alternative:
            # Mark multiples of pk, then unmark multiples of p_next.
            # Or just do it directly - for N=10^6, the total work is sum over p,k of N/(p^k) which is manageable.
            
            for m in range(pk, N + 1, pk):
                if m % p_next != 0:
                    P[m] *= pk
            pk = p_next
    
    # 3. Now P[n] is the powerful part of n for all n in [1, N].
    #    Compute score[n] = P[n] * P[n+1] for n in [MIN_N, N-1].
    #    Find n that maximizes score[n].
    
    best_n = MIN_N
    best_val = 0
    
    for n in range(MIN_N, N):
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
