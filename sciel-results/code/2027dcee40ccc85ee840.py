"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import isqrt

def construct():
    N = 123457
    
    # We want to find a, b > 0, a + b = N, minimizing max(P(a), P(b))
    # Strategy: Iterate primes p in increasing order. Maintain the set of p-smooth numbers.
    # If we find a pair (a, b) such that a+b=N and both are p-smooth, return them.
    # The first such p found guarantees the minimal max prime factor.
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    # We only need primes up to N/2 roughly, as one number is at least 1.
    # Actually, P(a) can be up to N-1. But if max(P(a),P(b)) is small, 
    # the primes involved are small. 
    # However, to be safe, we generate primes up to N.
    primes = sieve(N)
    
    # Initialize smooth set with 1
    smooth_set = {1}
    
    for idx, p in enumerate(primes):
        # Expand smooth_set to include numbers formed by multiplying existing smooth numbers by p
        # We need to be careful to only add numbers that are <= N
        new_adds = []
        # Iterate over a snapshot of the current set to avoid modifying while iterating
        # But we can be smarter: we only need to multiply by p^k
        # The current smooth_set contains all numbers whose prime factors are <= previous prime.
        # We want to add all numbers of the form s * p^k <= N where s is in smooth_set.
        
        # To ensure we don't miss combinations, we iterate through the current set.
        # Since p is the largest prime factor allowed now, any new number must have p as a factor.
        # So new number = s * p^k.
        
        # Optimization: Instead of iterating all s, we can just check which s * p^k are new.
        # But simple iteration is fine for small N.
        
        # We must iterate over the set as it was BEFORE adding new multiples of p?
        # No, if we add s*p, then (s*p)*p = s*p^2 is also valid.
        # So we should iterate over the original set, and for each s, add s*p, s*p^2, etc.
        
        original_size = len(smooth_set)
        # We need to iterate over the elements that were present before we started adding p-multiples?
        # Actually, if we add s*p, and then later in the same loop we process s*p, we might add s*p^2.
        # But if we just loop `for s in list(smooth_set)`, the list is a snapshot.
        # So if s=1, we add p, p^2, ...
        # If s=p (which was just added), it's not in the snapshot list if we took it at start.
        # So we must ensure that for each base s, we add all powers.
        
        # Let's take a snapshot of the current smooth_set
        current_smooths = list(smooth_set)
        
        for s in current_smooths:
            prod = s
            while prod <= N // p:
                prod *= p
                if prod > N:
                    break
                if prod not in smooth_set:
                    smooth_set.add(prod)
        
        # Check if any pair sums to N
        # We can optimize this check.
        # Instead of iterating all a in smooth_set, we can just check.
        # Since the set grows, this check is O(|S|).
        
        found = False
        # To make it faster, we can iterate over the smaller half?
        # Or just iterate all.
        for a in smooth_set:
            if a >= N:
                continue
            b = N - a
            if b in smooth_set:
                return [a, b]
        
    # Fallback
    return [1, N-1]
# EVOLVE-BLOCK-END
