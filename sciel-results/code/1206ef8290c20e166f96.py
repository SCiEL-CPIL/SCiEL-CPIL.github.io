"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import isqrt

def construct():
    N = 123457
    
    # We want to find a, b > 0, a + b = N, minimizing max(P(a), P(b))
    # Strategy: for increasing B, generate all B-smooth numbers <= N,
    # then check if any pair sums to N.
    
    # First, get all primes up to N
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(N)
    
    # Generate B-smooth numbers for increasing B
    # Start with B=2, then B=3, B=5, etc.
    
    for idx in range(len(primes)):
        B = primes[idx]
        # Generate all B-smooth numbers <= N
        # A number is B-smooth if all its prime factors are <= B
        smooth = {1}
        # Use a set and expand
        # For efficiency, generate by multiplying
        # We'll use a BFS/DFS approach
        smooth_set = set([1])
        # Add multiples
        for p in primes[:idx+1]:
            new_adds = set()
            for s in list(smooth_set):
                prod = s
                while prod <= N // p:
                    prod *= p
                    if prod > N:
                        break
                    if prod not in smooth_set:
                        new_adds.add(prod)
            smooth_set.update(new_adds)
        
        # Now check if any a in smooth_set has N - a also in smooth_set
        # and both > 0
        best_pair = None
        for a in smooth_set:
            if a < 1 or a >= N:
                continue
            b = N - a
            if b < 1:
                continue
            if b in smooth_set:
                # Both are B-smooth, so max(P(a), P(b)) <= B
                best_pair = (a, b)
                break
        
        if best_pair:
            return [best_pair[0], best_pair[1]]
    
    # Fallback: return the seed answer
    return [1, 123456]
# EVOLVE-BLOCK-END
