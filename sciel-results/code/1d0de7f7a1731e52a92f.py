"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from itertools import combinations

P = {'n': 256}

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _prime_factors(n):
    """Return set of prime factors of n."""
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

def _violates(S, a, b, c):
    """Check if a divides b*c."""
    return (b * c) % a == 0

def _check_valid(S):
    """Check if set S is valid: for all distinct a,b,c in S, a does not divide b*c."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = S_list[i], S_list[j], S_list[k]
                if (b * c) % a == 0:
                    return False
                if (a * c) % b == 0:
                    return False
                if (a * b) % c == 0:
                    return False
    return True

def construct():
    deadline = time.time() + 50  # 50s safety margin
    
    n_max = P['n']
    
    # Strategy: Start with all primes, then try to add composites greedily
    # A composite a can be added if for all pairs (b,c) in current set,
    # a does not divide b*c, and for all b in current set, b does not divide a*c for any c,
    # etc. Actually we need: for all distinct a,b,c in new set, none divides product of other two.
    
    # Let's think about which composites are "safe":
    # If a = p^2 (square of prime p), and p is in the set, then we need to check:
    #   Does p^2 divide p * c for some c? Only if p divides c.
    #   So if no other member is divisible by p, p^2 is safe to add alongside p.
    
    # Better approach: greedy by size, trying to add numbers that don't create violations.
    
    # First, get all primes
    primes = [x for x in range(2, n_max + 1) if _is_prime(x)]
    
    # Start with all primes - this is always valid since for distinct primes a,b,c,
    # a cannot divide b*c (a is prime and doesn't divide b or c)
    current = set(primes)
    
    # Now try to add composites in increasing order
    composites = [x for x in range(2, n_max + 1) if not _is_prime(x)]
    
    # For each composite, check if adding it maintains validity
    # To be efficient, we only need to check triples involving the new element
    def can_add(candidate, S):
        """Check if adding candidate to S maintains validity."""
        S_list = list(S)
        # Check triples where candidate is one of the three
        # Case 1: candidate divides b*c for some b,c in S (b != c, both != candidate)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                b, c = S_list[i], S_list[j]
                if (b * c) % candidate == 0:
                    return False
        # Case 2: b divides candidate*c for some b,c in S (c != b, both != candidate)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                b, c = S_list[i], S_list[j]
                if (candidate * c) % b == 0:
                    return False
                if (candidate * b) % c == 0:
                    return False
        return True
    
    # Try adding composites from smallest to largest
    for comp in composites:
        if time.time() > deadline:
            break
        if can_add(comp, current):
            current.add(comp)
    
    # Now try a second pass with larger composites (in case order matters)
    # Try adding in different orders
    improved = False
    for comp in reversed(composites):
        if time.time() > deadline:
            break
        if comp not in current and can_add(comp, current):
            current.add(comp)
            improved = True
    
    # Try another pass forward
    for comp in composites:
        if time.time() > deadline:
            break
        if comp not in current and can_add(comp, current):
            current.add(comp)
    
    # Try to improve by removing some elements and adding others
    # For now, return what we have
    result = sorted(current)
    
    # Verify validity
    if not _check_valid(set(result)):
        # Fallback to primes only
        result = primes
    
    # Ensure we return Python ints in a list
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
