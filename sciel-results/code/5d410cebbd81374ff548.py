"""Rewritten program: maximize number of distinct integers in 2..40 with distinct subset products."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}

# Precompute primes in 2..40
def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True

PRIMES = [a for a in range(2, 41) if _is_prime(a)]
# Primes: 2,3,5,7,11,13,17,19,23,29,31,37 -> 12 primes

def _check_distinct_subset_products(S):
    """
    Check if all 2^k subset products of S are distinct.
    S is a list of integers.
    """
    k = len(S)
    if k == 0:
        return True
    products = set()
    products.add(1)
    for x in S:
        new_products = set()
        for p in products:
            np = p * x
            if np in products or np in new_products:
                return False
            new_products.add(np)
        products.update(new_products)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before 60s limit
    
    # Try to find largest valid set
    # Start with primes (12 elements) and try to add composites
    base = list(PRIMES)  # 12 primes
    
    # Verify base is valid
    if not _check_distinct_subset_products(base):
        # Fallback: just return primes that work
        return base
    
    # Try to add composites one by one
    composites = [a for a in range(2, 41) if not _is_prime(a)]
    # Sort composites by some heuristic - maybe smaller ones first or by number of prime factors
    def prime_factor_count(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    composites.sort(key=lambda x: (prime_factor_count(x), x))
    
    best = list(base)
    
    # Greedy: try to add each composite if it keeps the set valid
    for c in composites:
        candidate = best + [c]
        if _check_distinct_subset_products(candidate):
            best = candidate
    
    # Try different orderings of composites
    # Randomized restarts
    import random
    random.seed(42)
    
    for trial in range(200):
        if time.time() > deadline:
            break
        trial_comps = list(composites)
        random.shuffle(trial_comps)
        trial_best = list(base)
        for c in trial_comps:
            candidate = trial_best + [c]
            if _check_distinct_subset_products(candidate):
                trial_best = candidate
        if len(trial_best) > len(best):
            best = trial_best
    
    # Try local search: remove one element and add others
    for trial in range(100):
        if time.time() > deadline:
            break
        # Start from best
        current = list(best)
        # Try to swap: remove one, add two
        improved = True
        while improved:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    return [int(x) for x in best]
                without_i = [x for j, x in enumerate(current) if j != i]
                for c1 in composites:
                    if c1 in current:
                        continue
                    for c2 in composites:
                        if c2 in current or c2 <= c1:
                            continue
                        candidate = without_i + [c1, c2]
                        if len(candidate) > len(current):
                            if _check_distinct_subset_products(candidate):
                                current = candidate
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        if len(current) > len(best):
            best = current
    
    # Final: ensure we return Python ints in a list
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
