"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using recursive prime-based generation."""
    max_p = isqrt(limit)
    if max_p < 2:
        return [1]
    
    # Sieve primes up to max_p
    is_prime = [True] * (max_p + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(max_p) + 1):
        if is_prime[i]:
            for j in range(i * i, max_p + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, max_p + 1) if is_prime[i]]
    
    result = set([1])
    
    def gen(idx, current):
        if current > limit:
            return
        if idx >= len(primes):
            if current > 1:
                result.add(current)
            return
        p = primes[idx]
        # Option 1: don't use prime p
        gen(idx + 1, current)
        # Option 2: use p^2, p^3, p^4, ...
        val = current * p * p
        while val <= limit:
            result.add(val)
            gen(idx + 1, val)
            val *= p
    
    gen(0, 1)
    return sorted(result)

def construct():
    start_time = time.time()
    
    # Set a limit that allows O(M^2) counting within time budget.
    # M ~ 0.64 * sqrt(L)
    # For L = 2 * 10^9, M ~ 28,000. M^2/2 ~ 4 * 10^8 ops.
    # This should take ~20-40 seconds in Python.
    L = 2_000_000_000
    
    powerful = _generate_powerful(L)
    n_pow = len(powerful)
    
    # Count pairs (a, b) with a <= b, a + b = N
    sum_counts = {}
    p_list = powerful
    
    for i in range(n_pow):
        a = p_list[i]
        for j in range(i, n_pow):
            b = p_list[j]
            s = a + b
            if s in sum_counts:
                sum_counts[s] += 1
            else:
                sum_counts[s] = 1
        
        # Optional: time check
        if (i & 0xFF) == 0:
            if time.time() - start_time > 130:
                break

    # Find the N with the most pairs
    if not sum_counts:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    best_N = max(sum_counts, key=sum_counts.get)
    best_count = sum_counts[best_N]
    
    # Collect all pairs for best_N
    p_set = set(powerful)
    pairs = []
    for a in powerful:
        b = best_N - a
        if b >= a and b in p_set:
            pairs.append([a, b])
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
