import time
import sys
from itertools import combinations

# EVOLVE-BLOCK-START

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

def _prime_factor_exponents(n):
    """Return dict {prime: exponent} for n >= 2."""
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

def _check_distinct_subset_products(nums):
    """Check all 2^k subset products are distinct."""
    k = len(nums)
    if k == 0:
        return True
    products = {1}
    for x in nums:
        # For each existing product, multiply by x
        new_products = set()
        for p in products:
            new_products.add(p * x)
        # Check if any new product already existed
        # But actually we need to check ALL products are distinct
        # Let's just build the full set and check length
        pass
    # Better approach: build all products
    all_products = [1]
    for x in nums:
        all_products = all_products + [p * x for p in all_products]
    return len(set(all_products)) == len(all_products)

def _check_distinct_subset_products_fast(nums):
    """Faster check using incremental set."""
    products = {1}
    for x in nums:
        new = set()
        for p in products:
            new.add(p * x)
        # If new product collides with existing, return False
        # Actually, we need to check if any element in new is in products
        # But wait - different subsets can produce same product
        # The issue: if x * p1 == x * p2 then p1 == p2 (in integers), so within the new set they're distinct
        # The collision can only be between new products and old products, or within new products
        # Within new products: x*p1 == x*p2 implies p1==p2, so distinct
        # Between new and old: x*p1 == p2 for some p2 in old products
        # This means p2 is divisible by x and p2/x is in old products
        # Simpler: just check intersection
        if new & products:
            return False
        products |= new
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # All numbers 2..40
    candidates = list(range(2, 41))
    
    # Precompute prime factorizations
    factorizations = {}
    for n in candidates:
        factorizations[n] = _prime_factor_exponents(n)
    
    # Strategy: Backtracking search
    # Order: primes first (they're always safe), then composites
    primes = [n for n in candidates if _is_prime(n)]
    composites = [n for n in candidates if not _is_prime(n)]
    # Sort composites by "complexity" - more prime factors first might be better
    # Actually, let's sort by number of prime factors descending
    composites.sort(key=lambda x: -sum(factorizations[x].values()))
    
    ordered_candidates = primes + composites
    
    best_set = []
    best_size = 0
    
    # Initial: try all primes
    if _check_distinct_subset_products_fast(primes):
        best_set = list(primes)
        best_size = len(primes)
    
    # Now do backtracking to try to improve
    # We'll use a recursive approach with pruning
    def backtrack(index, current_set, current_products):
        nonlocal best_set, best_size
        nonlocal deadline
        
        if time.time() > deadline:
            return
        
        # Pruning
        remaining = len(ordered_candidates) - index
        if len(current_set) + remaining <= best_size:
            return
        
        if index >= len(ordered_candidates):
            if len(current_set) > best_size:
                best_set = list(current_set)
                best_size = len(current_set)
            return
        
        # Try including ordered_candidates[index]
        x = ordered_candidates[index]
        new_products = set()
        valid = True
        for p in current_products:
            np = p * x
            if np in current_products or np in new_products:
                valid = False
                break
            new_products.add(np)
        
        if valid:
            current_set.append(x)
            current_products |= new_products
            backtrack(index + 1, current_set, current_products)
            current_set.pop()
            current_products -= new_products
        
        # Try excluding
        backtrack(index + 1, current_set, current_products)
    
    # Start backtracking from the primes already included
    initial_products = {1}
    for p in primes:
        new = set()
        for prod in initial_products:
            new.add(prod * p)
        initial_products |= new
    
    backtrack(0, list(primes), initial_products)
    
    # Ensure we return Python ints in a list
    result = [int(x) for x in best_set]
    return result

# EVOLVE-BLOCK-END
