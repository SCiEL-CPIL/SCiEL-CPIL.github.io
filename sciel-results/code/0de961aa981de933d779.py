"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import log

def _generate_smooth_numbers(primes, limit):
    """Generate all numbers of form prod(p_i^e_i) <= limit."""
    results = set([1])
    for p in primes:
        new_results = set()
        for r in results:
            val = r
            while val <= limit:
                new_results.add(val)
                val *= p
        results |= new_results
    return [x for x in results if x <= limit]

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s margin
    
    N = 123457
    
    # Get primes up to N
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i in range(2, n + 1) if sieve[i]]
    
    primes = get_primes(N)
    
    # Try increasing bounds on largest prime factor
    # For bound B, generate all B-smooth numbers <= N
    # Check if any pair sums to N
    best_max_pf = float('inf')
    best_a = 1
    
    # Incrementally add primes and check
    # Start with smallest primes
    for idx in range(len(primes)):
        if time.time() > deadline:
            break
        B = primes[idx]
        
        # Generate all B-smooth numbers up to N
        smooth = _generate_smooth_numbers(primes[:idx+1], N)
        smooth_set = set(smooth)
        
        # Check if any a in smooth has N-a also in smooth
        found = False
        for a in smooth:
            if a >= N:
                break
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
        # Fallback: linear scan
        spf = list(range(N + 1))
        for i in range(2, int(N**0.5) + 1):
            if spf[i] == i:
                for j in range(i*i, N + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        
        def _lpf(n, spf):
            if n <= 1:
                return 1
            max_pf = 1
            while n > 1:
                p = spf[n]
                max_pf = max(max_pf, p)
                while n % p == 0:
                    n //= p
            return max_pf
        
        for a in range(1, N):
            if time.time() > deadline:
                break
            b = N - a
            m = max(_lpf(a, spf), _lpf(b, spf))
            if m < best_max_pf:
                best_max_pf = m
                best_a = a
    
    return [best_a, N - best_a]
# EVOLVE-BLOCK-END
