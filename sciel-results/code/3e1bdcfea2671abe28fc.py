"""Rewritten program: maximize set size under no-three-distinct-product-divisibility."""
import time
from math import isqrt, gcd
from itertools import combinations

P = {'n': 256}

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    N = P['n']
    
    # Precompute primes in [2, N]
    sieve = [True] * (N + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(N) + 1):
        if sieve[i]:
            for j in range(i*i, N+1, i):
                sieve[j] = False
    primes = [i for i in range(2, N+1) if sieve[i]]
    composites = [i for i in range(2, N+1) if not sieve[i]]
    
    # For a candidate set S, check validity: for all distinct a,b,c in S, a !| b*c
    def is_valid(S):
        S = list(S)
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i+1, n):
                b = S[j]
                for k in range(j+1, n):
                    c = S[k]
                    if (b * c) % a == 0:
                        return False
        return True
    
    # Greedy: start with all primes, then try to add composites one by one
    # A composite x can be added if for all pairs (b,c) in current set, x does not divide b*c
    # AND for all existing a, a does not divide x*b or x*c for any b,c... 
    # Actually we need: for all distinct a,b,c in new set, a !| b*c
    # When adding x: new triples are those involving x. So check:
    # 1. For all pairs (b,c) in old set: x !| b*c
    # 2. For all a in old set, and all b in old set (b != a): a !| x*b
    #    (the triple is (a, x, b) or (a, b, x) - but a must not divide x*b)
    
    def can_add(x, S):
        S_list = list(S)
        n = len(S_list)
        # Check x as the divisor: x !| b*c for all distinct b,c in S
        for i in range(n):
            for j in range(i+1, n):
                if (S_list[i] * S_list[j]) % x == 0:
                    return False
        # Check each a in S as divisor: a !| x*b for all b in S, b != a
        for i in range(n):
            a = S_list[i]
            for j in range(n):
                if i == j:
                    continue
                b = S_list[j]
                if (x * b) % a == 0:
                    return False
        return True
    
    # Start with all primes
    best_set = set(primes)
    
    # Try to add composites in order of "safety" - prefer those with fewest prime factors
    # Sort composites by number of prime factors (fewer = safer)
    def num_prime_factors(x):
        count = 0
        for p in primes:
            if p > x:
                break
            if x % p == 0:
                count += 1
                while x % p == 0:
                    x //= p
        return count
    
    composites_sorted = sorted(composites, key=lambda x: (num_prime_factors(x), x))
    
    for x in composites_sorted:
        if time.time() > deadline:
            break
        if can_add(x, best_set):
            best_set.add(x)
    
    # Try local search: remove one, try to add two others
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = list(best_set)
        for i in range(len(S_list)):
            if time.time() > deadline:
                break
            candidate = S_list[i]
            best_set.remove(candidate)
            added = 0
            for x in composites_sorted:
                if x in best_set:
                    continue
                if can_add(x, best_set):
                    best_set.add(x)
                    added += 1
                    if added >= 2:
                        break
            if added > 1:
                improved = True
            elif added == 1:
                # Keep it, but mark as improved since we tried
                pass
            else:
                best_set.add(candidate)
    
    result = sorted(best_set)
    return [int(x) for x in result]
