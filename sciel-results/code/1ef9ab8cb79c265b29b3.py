"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import log

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s margin
    
    N = 123457
    
    # Get primes up to N using sieve
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i in range(2, n + 1) if sieve[i]]
    
    primes = get_primes(N)
    
    best_max_pf = float('inf')
    best_a = 1
    
    # Incrementally build smooth number set
    # Start with {1}, then add powers of each prime one at a time
    smooth_set = set([1])
    
    for idx in range(len(primes)):
        if time.time() > deadline:
            break
        B = primes[idx]
        p = B
        
        # Add all multiples by powers of p to the smooth set
        # For each existing smooth number r, add r*p, r*p^2, ... <= N
        current_list = list(smooth_set)
        for r in current_list:
            val = r * p
            while val <= N:
                smooth_set.add(val)
                val *= p
        
        # Check if any a in smooth_set has N-a also in smooth_set
        found = False
        for a in smooth_set:
            if a >= N:
                continue
            b = N - a
            if b >= 1 and b in smooth_set:
                best_max_pf = B
                best_a = a
                found = True
                break
        
        if found:
            # Can't do better than this prime bound
            break
    
    if best_max_pf == float('inf'):
        # Fallback: build SPF and scan
        spf = list(range(N + 1))
        for i in range(2, int(N**0.5) + 1):
            if spf[i] == i:
                for j in range(i*i, N + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        
        # Precompute largest prime factor for all numbers
        lpf = [1] * (N + 1)
        for i in range(2, N + 1):
            if spf[i] == i:  # i is prime
                for j in range(i, N + 1, i):
                    lpf[j] = i  # largest prime factor of j is at least i
                    # Actually we need the true largest, so do it differently
        
        # Better: compute lpf properly
        lpf = [1] * (N + 1)
        for i in range(2, N + 1):
            if spf[i] == i:  # prime
                for j in range(i, N + 1, i):
                    lpf[j] = i
        
        for a in range(1, N):
            if time.time() > deadline:
                break
            b = N - a
            m = max(lpf[a], lpf[b])
            if m < best_max_pf:
                best_max_pf = m
                best_a = a
    
    return [best_a, N - best_a]
# EVOLVE-BLOCK-END
