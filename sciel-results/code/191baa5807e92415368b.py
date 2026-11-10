"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from math import isqrt

def _has_distinct_subset_products(nums):
    """Check if all 2^k subset products of nums are distinct."""
    k = len(nums)
    if k == 0:
        return True
    if k > 18:
        return False
    products = set()
    products.add(1)
    for x in nums:
        # For each existing product, multiply by x and check for collision
        new_products = []
        for p in products:
            np = p * x
            if np in products:
                return False
            new_products.append(np)
        products.update(new_products)
    return True

def _build_candidates():
    """Build candidate numbers from 2..40 with prime factorization info."""
    candidates = []
    for n in range(2, 41):
        # Factorize
        factors = {}
        m = n
        for p in range(2, isqrt(m) + 1):
            while m % p == 0:
                factors[p] = factors.get(p, 0) + 1
                m //= p
        if m > 1:
            factors[m] = factors.get(m, 0) + 1
        candidates.append((n, factors))
    return candidates

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    candidates = _build_candidates()
    
    # Strategy: Greedy with prime-power priority, then try to add mixed numbers
    # First pass: select prime powers with superincreasing exponents per prime
    # Group by prime
    prime_powers = {}
    for n, factors in candidates:
        if len(factors) == 1:
            p, e = list(factors.items())[0]
            if p not in prime_powers:
                prime_powers[p] = []
            prime_powers[p].append((n, e))
    
    # For each prime, select superincreasing exponents
    selected = set()
    for p, pows in sorted(prime_powers.items(), key=lambda x: -x[0][1]):
        # Sort by exponent
        pows.sort(key=lambda x: x[1])
        # Greedily pick superincreasing
        chosen = []
        current_sum = 0
        for n, e in pows:
            if e > current_sum:
                chosen.append(n)
                current_sum += e
        selected.update(chosen)
    
    # Second pass: try to add remaining candidates (mixed primes) greedily
    remaining = [n for n, f in candidates if n not in selected]
    # Sort by number of prime factors (fewer = less constraining)
    remaining.sort(key=lambda n: sum(1 for _, f in candidates if f[0] == n for p, e in f.items()))
    
    current_list = sorted(selected)
    for n in remaining:
        if time.time() > deadline:
            break
        test_list = current_list + [n]
        test_list.sort()
        if len(test_list) <= 18 and _has_distinct_subset_products(test_list):
            current_list = test_list
    
    # Third pass: try local search - swap one element for another
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current_set = set(current_list)
            for remove_n in list(current_list):
                if time.time() > deadline - 5:
                    break
                for add_n in range(2, 41):
                    if add_n in current_set:
                        continue
                    new_list = [x for x in current_list if x != remove_n] + [add_n]
                    new_list.sort()
                    if _has_distinct_subset_products(new_list):
                        # Check if this is an improvement (same size but different, or we track)
                        # For now, accept any valid swap to explore
                        current_list = new_list
                        improved = True
                        break
                if improved:
                    break
    
    result = sorted(current_list)
    # Final validation
    if _has_distinct_subset_products(result):
        return list(result)
    else:
        # Fallback: just the primes
        primes = [n for n in range(2, 41) if all(n % d for d in range(2, isqrt(n) + 1))]
        return list(primes)
# EVOLVE-BLOCK-END

