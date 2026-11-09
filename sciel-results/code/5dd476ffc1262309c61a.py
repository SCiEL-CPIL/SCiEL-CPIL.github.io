"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from math import isqrt

def _prime_factorize(n):
    """Return dict of prime -> exponent for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _subset_products_distinct(nums):
    """Check if all 2^k subset products of nums are distinct."""
    k = len(nums)
    if k == 0:
        return True
    products = set()
    # Use iterative approach: for each number, double the set of products
    products = {1}
    for a in nums:
        new_products = set()
        for p in products:
            new_products.add(p * a)
        if len(new_products) != len(products):
            return False  # collision detected
        products |= new_products
        # Early termination if too many products
        if len(products) > (1 << min(k, 20)):
            break
    return len(products) == (1 << k)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    # All integers 2..40
    candidates = list(range(2, 41))
    
    # Precompute prime factorizations
    factorizations = {n: _prime_factorize(n) for n in candidates}
    
    # Greedy: try to build a large set
    # Strategy: prioritize numbers with rare prime factors
    # Count how many candidates use each prime
    from collections import Counter
    prime_counts = Counter()
    for n in candidates:
        for p in factorizations[n]:
            prime_counts[p] += 1
    
    # Score each number: sum of 1/prime_counts[p] for its primes (rarity)
    def rarity_score(n):
        return sum(1.0 / prime_counts[p] for p in factorizations[n])
    
    # Sort by rarity (descending), then by value
    sorted_candidates = sorted(candidates, key=lambda n: (-rarity_score(n), n))
    
    best_set = []
    
    # Greedy construction
    for n in sorted_candidates:
        if time.time() > deadline:
            break
        test_set = best_set + [n]
        if _subset_products_distinct(test_set):
            best_set.append(n)
    
    # Try to improve: local search - try swapping
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            current_size = len(best_set)
            best_set_set = set(best_set)
            for n in candidates:
                if n in best_set_set:
                    continue
                if time.time() > deadline - 2.0:
                    break
                # Try adding n and removing some other element
                for i in range(len(best_set)):
                    if time.time() > deadline - 2.0:
                        break
                    candidate_set = [x for j, x in enumerate(best_set) if j != i] + [n]
                    if _subset_products_distinct(candidate_set):
                        best_set = candidate_set
                        best_set_set = set(best_set)
                        improved = True
                        break
                if improved:
                    break
    
    # Ensure we return Python ints in a list
    result = [int(x) for x in best_set]
    return result
# EVOLVE-BLOCK-END

