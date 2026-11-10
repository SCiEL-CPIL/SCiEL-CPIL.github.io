"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from functools import lru_cache

P = {'n': 256}

def _sieve_primes(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i in range(2, n+1) if sieve[i]]

def _prime_factors(n):
    """Return set of distinct prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _max_power_for_prime(p, limit):
    """Find the largest power of p that is <= limit."""
    max_power = 1
    current = p
    while current * p <= limit:
        current *= p
        max_power *= p
    return max_power

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    n = P['n']
    primes = _sieve_primes(n)
    
    # Step 1: Start with 1 and the largest prime power for each prime
    # This ensures pairwise coprime (each number has exactly one prime factor)
    initial_set = {1}
    for p in primes:
        initial_set.add(_max_power_for_prime(p, n))
    
    # Step 2: Local search - try to merge small coprime numbers into larger products
    # or split composite numbers if beneficial
    # Since all our numbers are prime powers, we can only merge numbers with disjoint prime factors
    # But all prime powers have disjoint prime factors by construction
    
    # The key insight: we can replace a set of prime powers {p1^a1, p2^a2, ...} with their product
    # if the product <= n, and the product is larger than the sum of the individual powers
    # Wait, that's not right. We want to MAXIMIZE the sum, so we want to keep numbers as large as possible.
    
    # Actually, let me reconsider. The constraint is that each number must be <= 256.
    # If we have two prime powers a and b where gcd(a,b)=1, we could either:
    # 1. Keep them separate: contribute a + b to the sum
    # 2. Merge them into a*b if a*b <= 256: contribute a*b to the sum
    # We should merge if a*b > a + b, i.e., (a-1)(b-1) > 1, which is true for a,b >= 2
    
    # So the strategy is: greedily merge small prime powers into larger products where possible
    # to maximize the sum.
    
    # Let's do a more careful local search:
    # Start with all prime powers, then repeatedly try to merge pairs if beneficial and possible
    
    current_set = set(initial_set)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Sort by value to try merging smaller ones first
        items = sorted([x for x in current_set if x > 1])
        for i in range(len(items)):
            if time.time() >= deadline:
                break
            for j in range(i+1, len(items)):
                a = items[i]
                b = items[j]
                if a * b <= n and a * b > a + b:
                    # Check if a and b are still in the set (they might have been removed)
                    if a in current_set and b in current_set:
                        current_set.remove(a)
                        current_set.remove(b)
                        current_set.add(a * b)
                        improved = True
                        items = sorted([x for x in current_set if x > 1])
                        break
            if improved:
                break
    
    # Step 3: Try to add 1 if not already present (it's always allowed and adds to sum)
    if 1 not in current_set:
        current_set.add(1)
    
    # Step 4: Try to add any remaining numbers that are coprime to all in the set
    # This is unlikely to help since we've already used all primes, but let's check
    # Actually, any new number must have prime factors not in the current set,
    # but we've used all primes, so no new numbers can be added.
    
    # Step 5: Try splitting composite numbers if it increases the sum
    # If we have a composite number c in the set, we could split it into factors
    # But c is a product of prime powers, and splitting would give us smaller numbers
    # which would decrease the sum, so this is never beneficial.
    
    # Convert to list and ensure all are Python ints
    result = [int(x) for x in sorted(current_set)]
    
    # Verify pairwise coprime
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                # This should not happen, but if it does, fall back to initial set
                result = [int(x) for x in sorted(initial_set)]
                break
    
    return result

# EVOLVE-BLOCK-END
