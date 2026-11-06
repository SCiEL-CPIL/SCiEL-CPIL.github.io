"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from functools import reduce

def _sieve_primes(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    N = 512
    primes = _sieve_primes(N - 1)
    
    # Initial set: all primes and 1
    # We store the set of chosen numbers
    chosen = set([1])
    for p in primes:
        chosen.add(p)
        
    # Precompute weights
    weights = {a: 1.0 / (N - a) for a in range(1, N)}
    
    # List of composites to consider, sorted by weight descending (i.e., a descending)
    composites = [a for a in range(1, N) if a not in primes and a != 1]
    composites.sort(key=lambda x: -x) # Descending a means descending weight
    
    improved = True
    while improved:
        improved = False
        # We need to re-evaluate composites because the set 'chosen' changes
        # To be efficient, we can just iterate through all composites not in chosen
        for c in composites:
            if c in chosen:
                continue
                
            # Find primes in 'chosen' that divide c
            dividing_primes = []
            for p in primes:
                if c % p == 0 and p in chosen:
                    dividing_primes.append(p)
            
            # Check if c is coprime to all other elements in chosen
            # Other elements = chosen - dividing_primes
            # Note: 1 is coprime to everything.
            valid = True
            for x in chosen:
                if x in dividing_primes:
                    continue
                if x == 1:
                    continue
                if math.gcd(c, x) != 1:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Calculate gain
            gain = weights[c] - sum(weights[p] for p in dividing_primes)
            
            if gain > 1e-12: # Small epsilon to avoid floating point issues
                # Perform swap
                for p in dividing_primes:
                    chosen.remove(p)
                chosen.add(c)
                improved = True
                # Break to restart the loop with the new set? 
                # Or continue? Restarting is safer to ensure we pick the best next move
                # But iterating through all might find multiple disjoint improvements.
                # Let's just mark improved and continue checking others in this pass?
                # If we change 'chosen', the validity of subsequent 'c' might change.
                # Example: c1 and c2 share prime p. 
                # If we process c1, we remove p. 
                # Then c2 might become valid (if it was blocked by p) or invalid (if it conflicts with c1).
                # Since we are iterating, we should probably restart or be careful.
                # Given N=512, restart is fast enough.
                break
        
    return list(chosen)
# EVOLVE-BLOCK-END
