import math
import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Sieve to precompute largest prime factor for all numbers up to N
    lp = list(range(N + 1))  # lp[i] will store the largest prime factor of i
    lp[0] = 0
    lp[1] = 1
    
    for i in range(2, int(math.isqrt(N)) + 1):
        if lp[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                lp[j] = i
    
    # For primes greater than sqrt(N), lp[i] is still i (initialized correctly)
    # For composites with a prime factor > sqrt(N), we need to fix them
    # Actually, the above sieve only marks multiples starting from i*i.
    # For a composite n = p * q where p <= q and p > sqrt(N), that's impossible since p*q > N.
    # So all composites have at least one prime factor <= sqrt(N), and the sieve above
    # correctly sets lp[n] to the largest prime factor found by the sieve.
    # Wait, no. The sieve sets lp[j] = i for each prime i dividing j.
    # Since we iterate i in increasing order, the LAST prime i that divides j will be
    # the largest prime factor. But we're overwriting, so lp[j] ends up being the
    # largest prime i (in increasing order) that divides j and i <= sqrt(N).
    # If j has a prime factor > sqrt(N), it's j itself (if j is prime) or we need
    # to handle it. Let me reconsider.
    
    # Better approach: for each number, find its largest prime factor directly.
    # Use a modified sieve: for each prime p, mark all multiples of p with lp = p.
    # Since primes are processed in increasing order, the last mark wins, giving
    # the largest prime factor.
    
    lp = [1] * (N + 1)
    lp[0] = 0
    lp[1] = 1
    
    for i in range(2, N + 1):
        if lp[i] == 1:  # i is prime (not marked by any smaller prime)
            for j in range(i, N + 1, i):
                lp[j] = i
    
    # Now lp[n] is the largest prime factor of n for all n
    # Find the pair (a, b) with a + b = N that minimizes max(lp[a], lp[b])
    
    best_max_pf = N + 1
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N // 2 + 1):
        b = N - a
        current_max = max(lp[a], lp[b])
        if current_max < best_max_pf:
            best_max_pf = current_max
            best_a = a
            best_b = b
    
    return [best_a, best_b]
