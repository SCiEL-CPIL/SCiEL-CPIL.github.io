"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt, gcd
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 50.0  # 10s safety margin before 60s limit
    
    N = 256
    
    # Sieve of Eratosthenes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    best = list(primes)
    best_size = len(primes)
    
    # Helper: check if adding x to set S is valid
    # For all distinct a, b in S: x must not divide a*b, and a must not divide x*b, and b must not divide x*a
    # Actually the condition is: for every THREE DISTINCT members a,b,c of the full set, a does not divide b*c.
    # So when adding x to S, we need to check all triples involving x and two from S.
    # Triple (x, a, b): x does not divide a*b
    # Triple (a, x, b): a does not divide x*b
    # Triple (a, b, x): b does not divide x*a (same as above by symmetry of a,b)
    # Wait, more carefully: for distinct a, b, c in the set, a does not divide b*c.
    # So for triple (x, a, b): x | a*b must be False
    # For triple (a, x, b): a | x*b must be False
    # For triple (b, a, x): b | a*x must be False -- same as (a, x, b) with a,b swapped
    # For triple (a, b, x): a | b*x must be False -- same as (b, a, x)
    # So we need: x does not divide a*b, and a does not divide x*b, and b does not divide x*a
    # But a | x*b and b | x*a are the same condition up to swapping a and b.
    # Actually no: a | x*b means a divides x*b. b | x*a means b divides x*a. These are different.
    # So we need all three: x | a*b == False, a | x*b == False, b | x*a == False
    
    def can_add(x, S):
        """Check if x can be added to set S while maintaining validity."""
        for a, b in combinations(S, 2):
            # Check triple (x, a, b): x must not divide a*b
            if (a * b) % x == 0:
                return False
            # Check triple (a, x, b): a must not divide x*b
            if (x * b) % a == 0:
                return False
            # Check triple (b, a, x): b must not divide a*x  (same as a | b*x which is b | a*x)
            if (a * x) % b == 0:
                return False
        return True
    
    # Composites in 2..N that are not in primes
    composites = [i for i in range(2, N + 1) if not is_prime[i]]
    
    # Try to add composites one by one (greedy)
    current = list(best)
    for c in composites:
        if time.time() > deadline:
            break
        if can_add(c, current):
            current.append(c)
    
    if len(current) > best_size:
        best = current
        best_size = len(current)
    
    # Try local search: remove one element, try to add multiple others
    # This is more complex; for now, also try a different greedy order
    # Sort composites by some heuristic (e.g., fewer divisors in the prime set)
    
    # Try swapping: for each prime p, see if removing p allows adding multiple composites
    for p in list(best):
        if time.time() > deadline:
            break
        trial = [x for x in best if x != p]
        added = 0
        for c in composites:
            if c == p:
                continue
            if can_add(c, trial):
                trial.append(c)
                added += 1
        if len(trial) > best_size:
            best = trial
            best_size = len(trial)
    
    # Return as list of Python ints
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
