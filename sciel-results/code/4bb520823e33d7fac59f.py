"""Stronger algorithm for large sets with distinct subset products."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

DEADLINE = 85.0  # seconds; evaluator kills at 90s


def _is_valid_distinct_products(nums):
    """Check that all 2^k subset products are distinct."""
    n = len(nums)
    k = n
    # Use incremental product tracking
    # For each subset, compute product. Use a set.
    # We can do this efficiently by iterating over masks.
    # But for k up to 18, 2^18 = 262144, which is fine.
    products = set()
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= nums[i]
        if prod in products:
            return False
        products.add(prod)
    return True


def _validate_incremental(current, candidate):
    """Check if adding candidate to current set preserves distinct subset products.
    
    This is equivalent to checking that no product of a subset of current
    equals any product of a subset of current multiplied by candidate,
    times a product of a subset of {candidate}.
    Actually, the full check is: all subset products of current + [candidate] are distinct.
    The new subset products are:
    - all old subset products of current
    - all old subset products of current multiplied by candidate
    We need:
    1. No two old products collide (already guaranteed)
    2. No two new products collide: old_prod1 * candidate == old_prod2 * candidate => old_prod1 == old_prod2 (ok)
    3. No old product equals a new product: old_prod == old_prod2 * candidate
       => old_prod / old_prod2 == candidate
       This means candidate is a ratio of two subset products of current.
    So we need to check that candidate is not equal to p1/p2 for any distinct subset products p1, p2 of current.
    Equivalently, for every subset product p of current, p * candidate is not another subset product of current.
    """
    n = len(current)
    k = n
    # Compute all subset products of current
    prods = []
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= current[i]
        prods.append(prod)
    prod_set = set(prods)
    # Check: for each p in prods, p * candidate not in prod_set
    for p in prods:
        if p * candidate in prod_set:
            return False
    return True


def construct():
    start_time = time.time()
    
    # Strategy: Greedy + local search
    # 1. Start with a good initial set (e.g., primes or superincreasing-like)
    # 2. Try to add numbers from 2..40 that maintain distinct subset products
    # 3. Use local search: try swapping or adding
    
    # First, get all primes in 2..40
    def is_prime(a):
        if a < 2:
            return False
        for d in range(2, int(a**0.5) + 1):
            if a % d == 0:
                return False
        return True
    
    candidates = list(range(2, 41))
    
    # Greedy: try to build a large set
    # Start empty, add numbers in an order that maximizes size
    # Try different orderings
    
    best_set = []
    
    # Deadline check
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # First, try greedy with primes first
    primes = [a for a in candidates if is_prime(a)]
    composites = [a for a in candidates if not is_prime(a)]
    
    # Try several greedy orderings
    orderings = [
        primes + composites,
        sorted(candidates),
        sorted(candidates, reverse=True),
    ]
    
    for ordering in orderings:
        if time_left() < 5.0:
            break
        current = []
        for a in ordering:
            if a in current:
                continue
            if time_left() < 3.0:
                break
            # Check if we can add a
            if _validate_incremental(current, a):
                current.append(a)
        if len(current) > len(best_set):
            best_set = current[:]
    
    # Local search: try to improve by swapping
    # For each element in best_set, try replacing with a non-member
    improved = True
    while improved and time_left() > 10.0:
        improved = False
        current_set = set(best_set)
        for i, elem in enumerate(best_set):
            if time_left() < 10.0:
                break
            for a in candidates:
                if a in current_set:
                    continue
                # Try replacing elem with a
                new_set = best_set[:i] + [a] + best_set[i+1:]
                # Check validity
                if _is_valid_distinct_products(new_set):
                    best_set = new_set
                    current_set = set(best_set)
                    improved = True
                    break
    
    # Also try: start from best_set, try adding any remaining candidate
    for a in candidates:
        if a in best_set:
            continue
        if time_left() < 5.0:
            break
        new_set = best_set + [a]
        if _validate_incremental(best_set, a):
            best_set = new_set
    
    # Final validation
    if not _is_valid_distinct_products(best_set):
        # Fallback: return primes
        return [a for a in primes]
    
    # Ensure all are Python int
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
