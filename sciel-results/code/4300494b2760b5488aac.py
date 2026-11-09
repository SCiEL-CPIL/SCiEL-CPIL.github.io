"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
import random

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

def _can_add_to(n, S):
    """Check if n can be added to set S (S is a set, n not in S)."""
    others = list(S)
    for i in range(len(others)):
        b = others[i]
        for j in range(i + 1, len(others)):
            c = others[j]
            if (b * c) % n == 0:
                return False
            if (n * c) % b == 0:
                return False
            if (n * b) % c == 0:
                return False
    return True

def construct():
    deadline = time.time() + 80
    N = 256
    
    primes = [a for a in range(2, N + 1) if _is_prime(a)]
    composites = [a for a in range(2, N + 1) if not _is_prime(a)]
    
    best = set(primes)
    best_size = len(best)
    
    # Try multiple greedy orderings
    orderings = [
        composites,
        sorted(composites, reverse=True),
    ]
    
    # Sort by number of distinct prime factors
    def num_distinct_pf(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    orderings.append(sorted(composites, key=num_distinct_pf))
    orderings.append(sorted(composites, key=lambda x: -num_distinct_pf(x)))
    
    rng = random.Random(42)
    for _ in range(50):
        c = composites[:]
        rng.shuffle(c)
        orderings.append(c)
    
    for order in orderings:
        if time.time() > deadline - 10:
            break
        S = set(primes)
        for n in order:
            if time.time() > deadline - 5:
                break
            if n not in S and _can_add_to(n, S):
                S.add(n)
        if len(S) > best_size:
            best = S
            best_size = len(S)
    
    # Local search: try swaps
    if time.time() < deadline - 15:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            S_list = sorted(best)
            removed = [x for x in range(2, N + 1) if x not in best]
            for i in range(len(S_list)):
                if time.time() > deadline - 10:
                    break
                x = S_list[i]
                S_without_x = best - {x}
                for r in removed:
                    if time.time() > deadline - 10:
                        break
                    if _can_add_to(r, S_without_x):
                        best = S_without_x | {r}
                        improved = True
                        break
                if improved:
                    break
    
    return sorted(best)
# EVOLVE-BLOCK-END
